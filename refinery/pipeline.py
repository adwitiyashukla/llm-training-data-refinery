import shutil
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from refinery import fetch, wet
from refinery.bloom import Bloom
from refinery.config import ConfigError, blocklist_bytes, fingerprint
from refinery.langid import LangId, from_header
from refinery.minhash import MinHasher
from refinery.pii import redact
from refinery.rules import Rules, parse_blocklist
from refinery.runner import Clock, clean_tmp, crash_point, peak_mb, promote, read_json, run_tasks, sha256, write_json
from refinery.store import CORPUS, DEDUPED, DROPS, FILTERED, LEDGER, Layout, Table
from refinery.text import count_words, hash64, line_hashes, text_hash, url_key
from refinery.wet import WetError

CRAWL_STEP = 10**11
SHARD_STEP = 10**6
_MODELS = {}


def doc_base(number, index):
    return number * CRAWL_STEP + index * SHARD_STEP


def _langid():
    if "lid" not in _MODELS:
        _MODELS["lid"] = LangId()
    return _MODELS["lid"]


def banned_words(cfg):
    try:
        return parse_blocklist(blocklist_bytes(cfg))
    except ValueError as e:
        raise ConfigError(f"[url] blocklist: {e}") from None


def _save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    promote(tmp, path)


def filter_shard(job):
    cfg, crawl, index, rel, banned = job
    layout = Layout(cfg.run.data_dir)
    clock = Clock()
    raw = layout.raw(crawl.id, index)
    if not raw.exists():
        fetch.download(cfg.fetch.base_url + rel, raw, cfg.fetch.timeout, cfg.fetch.retries)
    fetch_seconds = clock.seconds()
    digest, size = sha256(raw), raw.stat().st_size
    rules = Rules(cfg, banned, _langid())
    base = doc_base(crawl.number, index)
    batch = cfg.run.batch_docs
    kept = Table(layout.filtered(crawl.id, index), FILTERED, batch)
    ledger = Table(layout.ledger(crawl.id, f"filter-{index:05d}"), LEDGER, batch)
    line_drops = Counter()
    pages = chars = 0
    try:
        for rec, page in enumerate(wet.pages(raw)):
            if rec >= SHARD_STEP:
                raise ValueError(f"shard {index} has more than {SHARD_STEP} records")
            if rec == 3:
                crash_point(f"filter:{index}")
            doc_id = base + rec
            verdict = rules.judge(page.url, page.text)
            cc_lang = from_header(page.lang) if page.lang else ""
            ledger.add(doc_id, page.url, cc_lang, verdict.lang, verdict.score, len(page.text),
                       count_words(page.text), verdict.reason, verdict.value)
            pages += 1
            chars += len(page.text)
            if verdict.reason:
                continue
            line_drops.update(why for _, why in verdict.removed)
            text = verdict.text
            kept.add(doc_id, page.url, page.date, text, count_words(text), verdict.score,
                     hash64(url_key(page.url)), text_hash(text), line_hashes(text))
    except WetError as e:
        raw.unlink(missing_ok=True)
        raise WetError(f"{e}; deleted the local copy, run again to download it fresh") from None
    result = {
        "pages": pages,
        "kept": kept.close(),
        "chars": chars,
        "bytes": size,
        "sha256": digest,
        "path": rel,
        "seconds": clock.seconds(),
        "fetch_seconds": fetch_seconds,
        "line_drops": dict(sorted(line_drops.items())),
        "peak_mb": peak_mb(),
    }
    ledger.close()
    write_json(layout.done(crawl.id, "filter", index), result)
    if not cfg.run.keep_raw:
        raw.unlink()
    return result


def minhash_shard(job):
    cfg, crawl, index = job
    layout = Layout(cfg.run.data_dir)
    clock = Clock()
    hasher = MinHasher(cfg.dedup.shingle_words, cfg.dedup.bands, cfg.dedup.rows, cfg.run.seed)
    source = pq.ParquetFile(layout.filtered(crawl.id, index))
    parts = {"ids": [], "url": [], "text": [], "sig": []}
    for batch in source.iter_batches(batch_size=cfg.run.batch_docs,
                                     columns=["doc_id", "text", "url_hash", "text_hash"]):
        sets = [hasher.shingles(t) for t in batch.column("text").to_pylist()]
        parts["sig"].append(hasher.signatures(sets))
        parts["ids"].append(batch.column("doc_id").to_numpy())
        parts["url"].append(batch.column("url_hash").to_numpy())
        parts["text"].append(batch.column("text_hash").to_numpy())
    width = cfg.dedup.bands * cfg.dedup.rows
    arrays = {name: np.concatenate(chunks) if chunks else np.zeros(0, np.int64) for name, chunks in parts.items()}
    if not parts["sig"]:
        arrays["sig"] = np.zeros((0, width), np.uint32)
        arrays["url"] = arrays["text"] = np.zeros(0, np.uint64)
    source.close()
    _save_npz(layout.sigs(crawl.id, index), keys=hasher.band_keys(arrays["sig"]), **arrays)
    result = {"docs": int(arrays["ids"].size), "seconds": clock.seconds(), "peak_mb": peak_mb()}
    write_json(layout.done(crawl.id, "minhash", index), result)
    return result


def lookup(keys, refs, values):
    if keys.size == 0 or values.size == 0:
        return np.full(values.size, -1, np.int64)
    pos = np.minimum(np.searchsorted(keys, values), keys.size - 1)
    return np.where(keys[pos] == values, refs[pos], -1)


def merge(keys, refs, new_keys, new_refs):
    all_keys = np.concatenate([keys, new_keys])
    all_refs = np.concatenate([refs, new_refs])
    uniq, first = np.unique(all_keys, return_index=True)
    return uniq, all_refs[first]


def duplicates(values, ids, keys, refs):
    prior = lookup(keys, refs, values)
    if values.size == 0:
        return np.zeros(0, bool), prior
    _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    earlier = ids[first[inverse]]
    return (prior >= 0) | (earlier != ids), np.where(prior >= 0, prior, earlier)


def union_find(size, left, right):
    parent = list(range(size))
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a < b:
            parent[b] = a
        elif b < a:
            parent[a] = b
    roots = np.array(parent, np.int64)
    while True:
        nxt = roots[roots]
        if np.array_equal(nxt, roots):
            return roots
        roots = nxt


def candidate_edges(keys, members):
    if members.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    flat = keys[members].ravel()
    owner = np.repeat(members.astype(np.int64), keys.shape[1])
    order = np.lexsort((owner, flat))
    flat, owner = flat[order], owner[order]
    start = np.ones(flat.size, bool)
    start[1:] = flat[1:] != flat[:-1]
    head = owner[np.maximum.accumulate(np.where(start, np.arange(flat.size), 0))]
    span = np.int64(keys.shape[0])
    pairs = np.unique(head[~start] * span + owner[~start])
    return pairs // span, pairs % span


def similarity(sig, left, right):
    out = np.empty(left.size)
    for start in range(0, left.size, 100_000):
        stop = start + 100_000
        out[start:stop] = (sig[left[start:stop]] == sig[right[start:stop]]).mean(axis=1)
    return out


def _array(state, name, dtype):
    path = state.path(name)
    return np.load(path) if path else np.zeros(0, dtype)


def load_sigs(layout, crawl, shards):
    parts = [np.load(layout.sigs(crawl.id, index)) for index, _ in shards]
    out = {name: np.concatenate([p[name] for p in parts]) for name in ("ids", "url", "text", "sig", "keys")}
    for p in parts:
        p.close()
    return out


def prior_matches(cfg, state, keys, sig):
    size = keys.shape[0]
    ref = np.full(size, -1, np.int64)
    sims = np.zeros(size)
    if not state.path("prior_keys"):
        return ref, sims
    prior_keys, prior_ids = _array(state, "prior_keys", np.uint64), _array(state, "prior_ids", np.int64)
    floor = cfg.dedup.min_similarity
    if floor > 0:
        sig_ids, prior_sig = _array(state, "prior_sig_ids", np.int64), _array(state, "prior_sig", np.uint32)
    for band in range(keys.shape[1]):
        found = lookup(prior_keys, prior_ids, keys[:, band])
        fresh = (ref < 0) & (found >= 0)
        if floor > 0 and fresh.any():
            rows = np.flatnonzero(fresh)
            values = (sig[rows] == prior_sig[np.searchsorted(sig_ids, found[rows])]).mean(axis=1)
            fresh[rows[values < floor]] = False
            sims[rows[values >= floor]] = values[values >= floor]
        ref = np.where(fresh, found, ref)
    return ref, sims


def cluster_stage(cfg, crawl, layout, state, shards, log):
    clock = Clock()
    info = state.crawl(crawl.id)
    data = load_sigs(layout, crawl, shards)
    ids, sig, keys = data["ids"], data["sig"], data["keys"]
    size = ids.size
    url_keys, url_refs = _array(state, "url_keys", np.uint64), _array(state, "url_refs", np.int64)
    text_keys, text_refs = _array(state, "text_keys", np.uint64), _array(state, "text_refs", np.int64)
    url_dup, url_ref = duplicates(data["url"], ids, url_keys, url_refs)
    rest = np.flatnonzero(~url_dup)
    exact_dup = np.zeros(size, bool)
    exact_ref = np.full(size, -1, np.int64)
    exact_dup[rest], exact_ref[rest] = duplicates(data["text"][rest], ids[rest], text_keys, text_refs)
    unique = ~url_dup & ~exact_dup
    prior_ref, prior_sim = prior_matches(cfg, state, keys, sig)
    prior_hit = unique & (prior_ref >= 0)
    alive = unique & ~prior_hit if cfg.dedup.cross_crawl else unique
    left, right = candidate_edges(keys, np.flatnonzero(alive))
    if cfg.dedup.min_similarity > 0:
        ok = similarity(sig, left, right) >= cfg.dedup.min_similarity
        left, right = left[ok], right[ok]
    roots = union_find(size, left, right)
    keep = alive & (roots == np.arange(size))
    near = np.flatnonzero(alive & ~keep)
    near_sim = similarity(sig, near, roots[near])
    drops = Table(layout.ledger(crawl.id, "cluster"), DROPS, cfg.run.batch_docs)
    dropped_prior = prior_hit & ~alive
    for i in range(size):
        if url_dup[i]:
            drops.add(int(ids[i]), "url_dup", 0.0, int(url_ref[i]))
        elif exact_dup[i]:
            drops.add(int(ids[i]), "exact_dup", 0.0, int(exact_ref[i]))
        elif dropped_prior[i]:
            drops.add(int(ids[i]), "near_dup_prior", float(prior_sim[i]), int(prior_ref[i]))
    for i, value in zip(near.tolist(), near_sim.tolist(), strict=True):
        drops.add(int(ids[i]), "near_dup", float(value), int(ids[roots[i]]))
    drops.close()
    sizes = np.bincount(roots[alive], minlength=size)
    grouped = sizes[sizes > 1]
    base = crawl.number * CRAWL_STEP
    stats = {
        "docs_in": int(size),
        "url_dup": int(url_dup.sum()),
        "exact_dup": int(exact_dup.sum()),
        "url_dup_prior": int((url_dup & (url_ref < base)).sum()),
        "exact_dup_prior": int((exact_dup & (exact_ref < base)).sum()),
        "prior_overlap": int(prior_hit.sum()),
        "near_dup_prior": int(dropped_prior.sum()),
        "near_dup": int(near.size),
        "kept": int(keep.sum()),
        "edges": int(left.size),
        "clusters": int(grouped.size),
        "largest_cluster": int(grouped.max()) if grouped.size else 0,
        "cluster_sizes": [[label, int(((grouped >= lo) & (grouped <= hi)).sum())]
                          for label, lo, hi in (("2", 2, 2), ("3", 3, 3), ("4-9", 4, 9), ("10-99", 10, 99),
                                                ("100+", 100, 10**12))],
    }
    keep_path = layout.keep(crawl.id)
    tmp = keep_path.with_name(keep_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, ids[keep])
    promote(tmp, keep_path)
    _save_npz(layout.clusters(crawl.id), ids=ids, roots=roots, alive=alive)
    files = {}
    url_keys, url_refs = merge(url_keys, url_refs, data["url"], ids)
    text_keys, text_refs = merge(text_keys, text_refs, data["text"][rest], ids[rest])
    prior_keys, prior_ids = merge(_array(state, "prior_keys", np.uint64), _array(state, "prior_ids", np.int64),
                                  keys[keep].ravel(), np.repeat(ids[keep], keys.shape[1]))
    arrays = {"url_keys": url_keys, "url_refs": url_refs, "text_keys": text_keys, "text_refs": text_refs,
              "prior_keys": prior_keys, "prior_ids": prior_ids}
    if cfg.dedup.min_similarity > 0:
        arrays["prior_sig_ids"] = np.concatenate([_array(state, "prior_sig_ids", np.int64), ids[keep]])
        arrays["prior_sig"] = np.concatenate([_array(state, "prior_sig", np.uint32).reshape(-1, sig.shape[1]),
                                              sig[keep]])
    for name, array in arrays.items():
        files[name] = state.fresh(name, ".npy")
        np.save(files[name], array)
    crash_point("cluster")
    stats["seconds"] = clock.seconds()
    stats["peak_mb"] = peak_mb()
    info["cluster"] = stats
    state.commit(files, {crawl.id: info})
    log(f"cluster  {crawl.id}  docs {size:,}  url dups {stats['url_dup']:,}  exact dups {stats['exact_dup']:,}  "
        f"near dups {stats['near_dup']:,}  earlier crawls {stats['near_dup_prior']:,}  kept {stats['kept']:,}  "
        f"{stats['seconds']:.1f}s")


def _first(values):
    out = np.zeros(values.size, bool)
    if values.size:
        out[np.unique(values, return_index=True)[1]] = True
    return out


def lines_stage(cfg, crawl, layout, state, shards, log):
    info = state.crawl(crawl.id)
    bloom_path = state.path("bloom")
    if bloom_path:
        bloom = Bloom(np.load(bloom_path), state.head["bloom_hashes"])
    else:
        bloom = Bloom.empty(cfg.dedup.bloom_capacity, cfg.dedup.bloom_fpr)
    if info.get("lines_next", 0) == 0:
        info["lines_next"] = 0
        info["lines_exact"] = bloom_path is None
    keep = np.load(layout.keep(crawl.id))
    for pos in range(info["lines_next"], len(shards)):
        index = shards[pos][0]
        clock = Clock()
        table = pq.read_table(layout.filtered(crawl.id, index), columns=["doc_id", "url", "date", "text", "lines"])
        ids = table["doc_id"].to_numpy()
        alive = np.isin(ids, keep)
        column = table["lines"].combine_chunks()
        counts = np.diff(column.offsets.to_numpy())
        flat = column.values.to_numpy()[np.repeat(alive, counts)]
        nonzero = flat != 0
        keys = flat[nonzero]
        hits = bloom.contains(keys)
        seen = np.zeros(flat.size, bool)
        seen[nonzero] = hits | ~_first(keys)
        bloom.add(np.unique(keys))
        if info["lines_exact"]:
            _save_npz(layout.lines(crawl.id, index), keys=keys, hits=hits)
        texts, urls, dates = table["text"].to_pylist(), table["url"].to_pylist(), table["date"].to_pylist()
        out = Table(layout.deduped(crawl.id, index), DEDUPED, cfg.run.batch_docs)
        drops = Table(layout.ledger(crawl.id, f"lines-{index:05d}"), DROPS, cfg.run.batch_docs)
        stats = Counter()
        cursor = 0
        for i in np.flatnonzero(alive).tolist():
            mask = seen[cursor : cursor + counts[i]]
            cursor += counts[i]
            lines = texts[i].split("\n")
            removed = int(mask.sum())
            stats["lines"] += len(lines)
            if removed:
                lines = [line for line, gone in zip(lines, mask, strict=True) if not gone]
            text = "\n".join(lines)
            words = count_words(text)
            if words < cfg.dedup.min_words:
                drops.add(int(ids[i]), "line_dup", float(words), -1)
                stats["line_dup"] += 1
                stats["lines_in_dropped"] += removed
                continue
            stats["lines_removed"] += removed
            out.add(int(ids[i]), urls[i], dates[i], text, words, removed)
        stats["docs_in"] = int(alive.sum())
        stats["docs_out"] = out.close()
        drops.close()
        crash_point(f"lines:{index}")
        bloom_file = state.fresh("bloom", ".npy")
        np.save(bloom_file, bloom.bits)
        state.head["bloom_hashes"] = bloom.hashes
        stats["seconds"] = clock.seconds()
        write_json(layout.done(crawl.id, "lines", index), dict(stats))
        info["lines_next"] = pos + 1
        state.commit({"bloom": bloom_file}, {crawl.id: info})
        log(f"lines    {crawl.id}  shard {pos + 1}/{len(shards)}  docs {stats['docs_in']:,}  "
            f"lines removed {stats['lines_removed']:,}  {stats['seconds']:.1f}s")


def publish_shard(job):
    cfg, crawl, index = job
    layout = Layout(cfg.run.data_dir)
    clock = Clock()
    table = pq.read_table(layout.deduped(crawl.id, index))
    ids = table["doc_id"].to_numpy().tolist()
    texts, urls, dates = table["text"].to_pylist(), table["url"].to_pylist(), table["date"].to_pylist()
    out = Table(layout.corpus(crawl.id, index), CORPUS, cfg.run.batch_docs)
    totals = Counter()
    for n, doc_id in enumerate(ids):
        if n == 3:
            crash_point(f"publish:{index}")
        text, found = redact(texts[n], cfg.pii.emails, cfg.pii.ips, cfg.pii.phones)
        words = count_words(text)
        out.add(doc_id, urls[n], crawl.id, dates[n], text, words, found["emails"], found["ips"], found["phones"])
        totals.update(found)
        totals["words"] += words
        totals["chars"] += len(text)
    result = {"emails": 0, "ips": 0, "phones": 0, **totals}
    result["docs"] = out.close()
    path = layout.corpus(crawl.id, index)
    result.update(bytes=path.stat().st_size, sha256=sha256(path), seconds=clock.seconds(), peak_mb=peak_mb())
    write_json(layout.done(crawl.id, "publish", index), result)
    return result


def check_state(cfg, state):
    done = sorted(state.head["crawls"].items(), key=lambda kv: kv[1]["number"])
    if len(done) > len(cfg.crawls):
        raise ConfigError("the data folder has more crawls than config.toml lists; use a new data_dir")
    for (crawl_id, info), crawl in zip(done, cfg.crawls, strict=False):
        if crawl_id != crawl.id or info["number"] != crawl.number:
            raise ConfigError(f"config.toml lists {crawl.id} where the data folder has {crawl_id}; "
                              "crawls can only be appended")
        if info["fingerprint"] != fingerprint(cfg, crawl):
            raise ConfigError(f"settings changed since {crawl.id} was processed; use a new data_dir "
                              "or restore the old config.toml")
    unfinished = [crawl_id for crawl_id, info in done if not info.get("done")]
    if len(unfinished) > 1:
        raise ConfigError(f"more than one unfinished crawl in the data folder: {unfinished}")


def _parallel(stage, fn, jobs, workers, log, crawl, layout):
    pending = [job for job in jobs if not layout.done(crawl.id, stage, job[2]).exists()]
    total, finished = len(jobs), len(jobs) - len(pending)
    clock = Clock()

    def on_done(job, result):
        nonlocal finished
        finished += 1
        eta = clock.seconds() / max(1, finished - (total - len(pending))) * (total - finished)
        if "pages" in result:
            extra = f"pages {result['pages']:,}  kept {result['kept']:,}"
        else:
            extra = f"docs {result['docs']:,}"
        log(f"{stage:<8} {crawl.id}  shard {finished}/{total}  {extra}  {result['seconds']:.1f}s  "
            f"eta {eta / 60:.1f} min")

    run_tasks(fn, pending, workers, on_done)


def process_crawl(cfg, crawl, layout, state, banned, log):
    if state.crawl(crawl.id) is None:
        state.commit(crawls={crawl.id: {"number": crawl.number, "fingerprint": fingerprint(cfg, crawl),
                                        "seconds": {}}})
    clean_tmp(layout.root)
    paths = fetch.shard_list(cfg.fetch.base_url, crawl.id, layout.crawl(crawl.id), cfg.fetch.timeout,
                             cfg.fetch.retries)
    shards = fetch.pick(paths, crawl.shards)
    write_json(layout.crawl(crawl.id) / "shards.json", [[i, p] for i, p in shards])
    workers = cfg.run.workers
    steps = (
        ("filter", lambda: _parallel("filter", filter_shard, [(cfg, crawl, i, p, banned) for i, p in shards],
                                     workers, log, crawl, layout)),
        ("minhash", lambda: _parallel("minhash", minhash_shard, [(cfg, crawl, i) for i, _ in shards],
                                      workers, log, crawl, layout)),
        ("cluster", lambda: "cluster" in state.crawl(crawl.id)
            or cluster_stage(cfg, crawl, layout, state, shards, log)),
        ("lines", lambda: lines_stage(cfg, crawl, layout, state, shards, log)),
        ("publish", lambda: _parallel("publish", publish_shard, [(cfg, crawl, i) for i, _ in shards],
                                      workers, log, crawl, layout)),
    )
    for name, step in steps:
        clock = Clock()
        step()
        info = state.crawl(crawl.id)
        info["seconds"][name] = info["seconds"].get(name, 0.0) + clock.seconds()
        state.commit(crawls={crawl.id: info})
    return shards


def finish_crawl(cfg, crawl, layout, state):
    info = state.crawl(crawl.id)
    info["done"] = True
    state.commit(crawls={crawl.id: info})
    if not cfg.run.keep_work:
        shutil.rmtree(layout.work(crawl.id), ignore_errors=True)
    if not cfg.run.keep_raw:
        shutil.rmtree(layout.root / "raw" / crawl.id, ignore_errors=True)


def markers(layout, crawl_id, stage):
    folder = layout.work(crawl_id) / "done" / stage
    return {int(p.stem): read_json(p) for p in sorted(folder.glob("*.json"))} if folder.exists() else {}
