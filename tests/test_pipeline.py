import json
import os
import random
import subprocess
import sys

import duckdb
import pyarrow.parquet as pq
import pytest
from conftest import english, make_crawl, raw_settings, settings, write_config

from refinery import wet
from refinery.cli import main, run
from refinery.config import ConfigError, blocklist_bytes, fingerprint, parse
from refinery.pii import EMAIL_SUB
from refinery.pipeline import banned_words, filter_shard
from refinery.runner import read_json
from refinery.text import hash64

CRAWL = "CC-MAIN-2099-01"
LATER = "CC-MAIN-2099-02"
EXPECTED = {
    "foreign_fr": {"lang_score"},
    "foreign_de": {"lang_score"},
    "foreign_es": {"lang_score"},
    "spam": {"rep_dup_lines", "rep_dup_line_chars", "rep_dup_paragraphs"},
    "short": {"gopher_few_words", "lang_score"},
    "code": {"c4_curly_bracket"},
    "lorem": {"c4_lorem_ipsum"},
    "blocked": {"url_blocklist"},
    "exact": {"exact_dup"},
    "near": {"near_dup"},
}


def outcomes(data, crawl):
    db = duckdb.connect()
    folder = (data / "ledger" / crawl).as_posix()
    filt = dict(db.execute(f"SELECT url, reason FROM read_parquet('{folder}/filter-*.parquet')").fetchall())
    ids = dict(db.execute(f"SELECT doc_id, url FROM read_parquet('{folder}/filter-*.parquet')").fetchall())
    for name in ("cluster.parquet", "lines-*.parquet"):
        for doc_id, reason in db.execute(f"SELECT doc_id, reason FROM read_parquet('{folder}/{name}')").fetchall():
            filt[ids[doc_id]] = reason
    return filt


def corpus(data, crawl):
    return pq.read_table((data / "corpus" / crawl).as_posix()).to_pylist()


def digests(data, crawl):
    manifest = read_json(data / "reports" / crawl / "manifest.json")
    return [(f["file"], f["sha256"]) for f in manifest["corpus"] + manifest["ledger"]]


def test_planted_pages_end_where_expected(tmp_path, web):
    base, truth, _ = web
    cfg = settings(tmp_path, base, [(CRAWL, 3)])
    assert run(cfg) == 0
    data = tmp_path / "data"
    result = outcomes(data, CRAWL)
    assert len(result) == len(truth)
    for url, kind in truth.items():
        if kind in EXPECTED:
            assert result[url] in EXPECTED[kind], (kind, result[url])
    published = corpus(data, CRAWL)
    english = [u for u, k in truth.items() if k == "english"]
    assert sum(result[u] == "" for u in english) >= 0.9 * len(english)
    assert len(published) == sum(1 for r in result.values() if r == "")
    pii = [row for row in published if truth[row["url"]] == "pii"]
    assert pii and all(EMAIL_SUB in row["text"] and "8.8." not in row["text"] for row in pii)
    assert all(row["emails"] == 1 and row["ips"] == 1 and row["phones"] == 1 for row in pii)
    metrics = read_json(data / "reports" / CRAWL / "metrics.json")
    assert metrics["consistent"]
    assert metrics["published"]["docs"] == len(published)
    assert metrics["eval"]["bloom"]["exact_false_negatives"] == 0
    assert (data / "reports" / CRAWL / "card.md").exists()
    assert not (data / "work").joinpath(CRAWL).exists()
    assert not (data / "raw").joinpath(CRAWL).exists()


def test_same_output_for_any_worker_count(tmp_path, web):
    base, _, _ = web
    one = tmp_path / "one"
    three = tmp_path / "three"
    run(settings(one, base, [(CRAWL, 3)], workers=1))
    run(settings(three, base, [(CRAWL, 3)], workers=3))
    assert digests(one / "data", CRAWL) == digests(three / "data", CRAWL)


@pytest.mark.parametrize("point", ["filter:1", "lines:1", "cluster", "publish:2"])
@pytest.mark.parametrize("workers", [1, 2])
def test_resume_after_crash_gives_identical_output(tmp_path, web, point, workers):
    base, _, _ = web
    clean = tmp_path / "clean"
    run(settings(clean, base, [(CRAWL, 3)]))
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    config = write_config(crashed / "config.toml", raw_settings(crashed, base, [(CRAWL, 3)], workers=workers))
    env = {**os.environ, "REFINERY_CRASH": point}
    first = subprocess.run([sys.executable, "-m", "refinery", "run", "--config", str(config)], env=env,
                           capture_output=True, text=True)
    assert first.returncode != 0
    env.pop("REFINERY_CRASH")
    second = subprocess.run([sys.executable, "-m", "refinery", "run", "--config", str(config)], env=env,
                            capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert digests(crashed / "data", CRAWL) == digests(clean / "data", CRAWL)


def test_second_crawl_is_deduplicated_against_the_first(tmp_path, web):
    base, _, made = web
    later, _ = make_crawl(base, LATER, shards=2, per_shard=50, seed=11, earlier=made)
    first = settings(tmp_path, base, [(CRAWL, 3)])
    run(first)
    data = tmp_path / "data"
    before = digests(data, CRAWL)
    both = settings(tmp_path, base, [(CRAWL, 3), (LATER, 2)])
    run(both)
    assert digests(data, CRAWL) == before
    result = outcomes(data, LATER)
    copies = [u for u, k in later.items() if k == "copy_prior"]
    near = [u for u, k in later.items() if k == "near_prior"]
    assert copies and all(result[u] in {"exact_dup", "line_dup"} for u in copies)
    assert near and sum(result[u] in {"near_dup_prior", "line_dup"} for u in near) >= 0.8 * len(near)
    metrics = read_json(data / "reports" / LATER / "metrics.json")
    assert metrics["dedup"]["cluster"]["exact_dup_prior"] >= 1
    assert "exact_fpr" not in metrics["eval"]["bloom"]


def test_cross_crawl_switch_only_measures(tmp_path, web):
    base, _, made = web
    make_crawl(base, LATER, shards=2, per_shard=50, seed=11, earlier=made)
    run(settings(tmp_path, base, [(CRAWL, 3), (LATER, 2)], dedup__cross_crawl=False))
    metrics = read_json(tmp_path / "data" / "reports" / LATER / "metrics.json")
    assert metrics["dedup"]["cluster"]["near_dup_prior"] == 0
    assert metrics["dedup"]["cluster"]["prior_overlap"] > 0


def test_changed_settings_are_refused(tmp_path, web):
    base, _, _ = web
    run(settings(tmp_path, base, [(CRAWL, 3)]))
    with pytest.raises(ConfigError, match="settings changed"):
        run(settings(tmp_path, base, [(CRAWL, 3)], language__min_score=0.5))
    with pytest.raises(ConfigError, match="appended"):
        run(settings(tmp_path, base, [(LATER, 3)]))


def test_cli_status_and_explain(tmp_path, web, capsys):
    base, _, made = web
    config = write_config(tmp_path / "config.toml", raw_settings(tmp_path, base, [(CRAWL, 3)]))
    assert main(["status", "--config", str(config)]) == 0
    assert "not started" in capsys.readouterr().out
    assert main(["run", "--config", str(config)]) == 0
    assert main(["status", "--config", str(config)]) == 0
    assert f"{CRAWL}: done" in capsys.readouterr().out
    page = tmp_path / "page.txt"
    page.write_text(made[0], encoding="utf-8")
    assert main(["explain", str(page), "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "verdict: kept" in out and "FAIL" not in out
    assert main(["export", "--config", str(config), "--out", str(tmp_path / "space")]) == 0
    exported = read_json(tmp_path / "space" / f"metrics-{CRAWL}.json")
    assert exported == read_json(tmp_path / "data" / "reports" / CRAWL / "metrics.json")
    assert "https://" not in json.dumps(exported)
    assert read_json(tmp_path / "space" / "crawls.json") == [CRAWL]


def test_bad_config_exits_cleanly(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text('[run]\nworkers = 0\n[[crawls]]\nid = "nope"\nshards = 1\n', encoding="utf-8")
    assert main(["status", "--config", str(config)]) == 2
    err = capsys.readouterr().err
    assert "workers must be between" in err and "needs an id" in err


def test_config_validation_messages():
    with pytest.raises(ConfigError, match="unknown keys: speed"):
        parse({"run": {"speed": 3}, "crawls": [{"id": CRAWL, "shards": 1}]})
    with pytest.raises(ConfigError, match="listed twice"):
        parse({"crawls": [{"id": CRAWL, "shards": 1}, {"id": CRAWL, "shards": 2}]})
    with pytest.raises(ConfigError, match="list of 3 numbers"):
        parse({"crawls": [{"id": CRAWL, "shards": 1}], "gopher_repetition": {"top_ngram_char_frac": [0.1]}})
    cfg = parse({"crawls": [{"id": CRAWL, "shards": 1}]})
    assert cfg.dedup.bands == 14 and cfg.language.keep == ["en"]


def test_minhash_without_verification(tmp_path, web):
    base, truth, _ = web
    run(settings(tmp_path, base, [(CRAWL, 3)], dedup__min_similarity=0.0))
    result = outcomes(tmp_path / "data", CRAWL)
    near = [u for u, k in truth.items() if k == "near"]
    assert near and all(result[u] == "near_dup" for u in near)
    variants = read_json(tmp_path / "data" / "reports" / CRAWL / "metrics.json")["eval"]["lsh"]["variants"]
    assert [v["min_similarity"] for v in variants] == [0.0, 0.6, 0.7, 0.8]


def test_status_reports_an_unfinished_crawl(tmp_path, web, capsys):
    base, _, _ = web
    config = write_config(tmp_path / "config.toml", raw_settings(tmp_path, base, [(CRAWL, 3)]))
    env = {**os.environ, "REFINERY_CRASH": "cluster"}
    first = subprocess.run([sys.executable, "-m", "refinery", "run", "--config", str(config)], env=env,
                           capture_output=True, text=True)
    assert first.returncode != 0
    assert main(["status", "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "filter 3/3" in out and "minhash 3/3" in out and "cluster pending" in out and "lines 0/3" in out


def test_filter_marker_counts_c4_line_drops(tmp_path):
    rng = random.Random(3)
    text = "\n".join([english(rng, 6), "Read more", "Please enable JavaScript to see the comments below this post.",
                      english(rng, 2)])
    base = tmp_path / "web"
    rel = f"crawl-data/{CRAWL}/segments/0000/wet/part-00000.warc.wet.gz"
    (base / rel).parent.mkdir(parents=True)
    wet.write(base / rel, [("https://blog.example/post", text, "eng")])
    cfg = settings(tmp_path, base, [(CRAWL, 1)])
    result = filter_shard((cfg, cfg.crawls[0], 0, rel, frozenset()))
    assert result["pages"] == 1 and result["kept"] == 1
    assert result["line_drops"] == {"c4_javascript_line": 1, "c4_short_line": 1}


def test_bundled_blocklist_is_hashes_only():
    cfg = parse({"crawls": [{"id": CRAWL, "shards": 1}]})
    raw = blocklist_bytes(cfg).decode("ascii")
    lines = raw.split()
    assert len(lines) > 300 and all(len(line) == 16 and int(line, 16) >= 0 for line in lines)
    banned = banned_words(cfg)
    assert len(banned) == len(lines)
    assert not any(hash64(word) in banned for word in ("news", "library", "example", "com", "www", "blog"))
    off = parse({"crawls": [{"id": CRAWL, "shards": 1}], "url": {"enabled": False}})
    assert banned_words(off) == frozenset()


def test_blocklist_file_errors_and_fingerprint(tmp_path):
    missing = parse({"crawls": [{"id": CRAWL, "shards": 1}], "url": {"blocklist": str(tmp_path / "nope.txt")}})
    with pytest.raises(ConfigError, match="blocklist file not found"):
        banned_words(missing)
    bad = tmp_path / "bad.txt"
    bad.write_text("not-a-hash\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="bad blocklist line"):
        banned_words(parse({"crawls": [{"id": CRAWL, "shards": 1}], "url": {"blocklist": str(bad)}}))
    one = tmp_path / "one.txt"
    one.write_text(f"{hash64('blockedtoken'):016x}\n", encoding="utf-8")
    two = tmp_path / "two.txt"
    two.write_text(f"{hash64('blockedsite'):016x}\n", encoding="utf-8")
    first = parse({"crawls": [{"id": CRAWL, "shards": 1}], "url": {"blocklist": str(one)}})
    second = parse({"crawls": [{"id": CRAWL, "shards": 1}], "url": {"blocklist": str(two)}})
    assert fingerprint(first, first.crawls[0]) != fingerprint(second, second.crawls[0])
