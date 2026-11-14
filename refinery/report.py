from collections import Counter

import duckdb

from refinery import __version__
from refinery.pipeline import markers
from refinery.runner import peak_mb, sha256, write_json

STEPS = (
    ("url blocklist", "reason = 'url_blocklist'"),
    ("language", "reason = 'lang_score'"),
    ("gopher repetition", "reason LIKE 'rep_%'"),
    ("gopher quality", "reason LIKE 'gopher_%'"),
    ("c4", "reason LIKE 'c4_%'"),
    ("fineweb", "reason LIKE 'fineweb_%'"),
)
DEDUP_STEPS = (
    ("url dedup", "url_dup"),
    ("exact dedup", "exact_dup"),
    ("near dup of earlier crawls", "near_dup_prior"),
    ("near dup", "near_dup"),
    ("line dedup", "line_dup"),
)


def _glob(layout, crawl_id, pattern):
    return (layout.root / "ledger" / crawl_id / pattern).as_posix()


def language(db, filters):
    row = db.execute(
        f"""
        SELECT count(*),
               count(*) FILTER (WHERE cc_lang <> ''),
               count(*) FILTER (WHERE cc_lang <> '' AND cc_lang = lang),
               count(*) FILTER (WHERE lang = 'en' AND reason <> 'lang_score'),
               count(*) FILTER (WHERE cc_lang = 'en'),
               count(*) FILTER (WHERE cc_lang = 'en' AND lang = 'en' AND reason <> 'lang_score')
        FROM read_parquet('{filters}')
        """
    ).fetchone()
    total, headed, agree, ft_en, cc_en, both = row
    top = db.execute(
        f"SELECT lang, count(*) AS n FROM read_parquet('{filters}') GROUP BY lang ORDER BY n DESC, lang LIMIT 10"
    ).fetchall()
    return {
        "pages": total,
        "with_cc_header": headed,
        "agreement": agree / headed if headed else 0.0,
        "fasttext_english": ft_en,
        "cc_english": cc_en,
        "both_english": both,
        "english_overlap_of_fasttext": both / ft_en if ft_en else 0.0,
        "english_overlap_of_cc": both / cc_en if cc_en else 0.0,
        "top": [[lang or "none", n] for lang, n in top],
    }


def build(cfg, crawl, layout, state, shards, evaluation):
    info = state.crawl(crawl.id)
    db = duckdb.connect()
    filters = _glob(layout, crawl.id, "filter-*.parquet")
    lines_glob = _glob(layout, crawl.id, "lines-*.parquet")
    cluster = _glob(layout, crawl.id, "cluster.parquet")
    pages, chars, words = db.execute(
        f"SELECT count(*), sum(chars), sum(words) FROM read_parquet('{filters}')"
    ).fetchone()
    reasons = dict(db.execute(
        f"SELECT reason, count(*) FROM read_parquet('{filters}') WHERE reason <> '' GROUP BY reason"
    ).fetchall())
    for pattern in (cluster, lines_glob):
        reasons.update(dict(db.execute(
            f"SELECT reason, count(*) FROM read_parquet('{pattern}') GROUP BY reason"
        ).fetchall()))
    remaining = pages
    funnel = [{"step": "pages read", "dropped": 0, "remaining": pages}]
    for name, where in STEPS:
        dropped = db.execute(f"SELECT count(*) FROM read_parquet('{filters}') WHERE {where}").fetchone()[0]
        remaining -= dropped
        funnel.append({"step": name, "dropped": dropped, "remaining": remaining})
    for name, reason in DEDUP_STEPS:
        dropped = reasons.get(reason, 0)
        remaining -= dropped
        funnel.append({"step": name, "dropped": dropped, "remaining": remaining})
    fetched = markers(layout, crawl.id, "filter")
    dedup = markers(layout, crawl.id, "lines")
    hashed = markers(layout, crawl.id, "minhash")
    published = markers(layout, crawl.id, "publish")
    docs = sum(m["docs"] for m in published.values())
    lines = sum(m.get("lines", 0) for m in dedup.values())
    removed = sum(m.get("lines_removed", 0) for m in dedup.values())
    seconds = dict(info["seconds"])
    wall = sum(seconds.values())
    worker_peak = max([m["peak_mb"] for group in (fetched, hashed, published) for m in group.values()] or [0.0])
    metrics = {
        "crawl": crawl.id,
        "shards": len(shards),
        "fingerprint": info["fingerprint"],
        "version": __version__,
        "input": {"pages": pages, "chars": int(chars or 0), "words": int(words or 0),
                  "gz_bytes": sum(m["bytes"] for m in fetched.values())},
        "funnel": funnel,
        "published": {
            "docs": docs,
            "words": sum(m.get("words", 0) for m in published.values()),
            "chars": sum(m.get("chars", 0) for m in published.values()),
            "parquet_bytes": sum(m["bytes"] for m in published.values()),
            "files": len(published),
        },
        "kept_share": docs / pages if pages else 0.0,
        "consistent": remaining == docs,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        "dedup": {
            "lines_seen": lines,
            "lines_removed": removed,
            "line_removed_share": removed / lines if lines else 0.0,
            "cluster": info.get("cluster", {}),
        },
        "c4_lines": dict(sorted(sum((Counter(m.get("line_drops", {})) for m in fetched.values()), Counter()).items())),
        "pii": {k: sum(m.get(k, 0) for m in published.values()) for k in ("emails", "ips", "phones")},
        "language": language(db, filters),
        "seconds": {**seconds, "total": wall,
                    "download": sum(m["fetch_seconds"] for m in fetched.values())},
        "pages_per_second": pages / wall if wall else 0.0,
        "peak_mb": {"worker": worker_peak, "main": peak_mb()},
        "workers": cfg.run.workers,
        "eval": evaluation,
    }
    db.close()
    folder = layout.report(crawl.id)
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "manifest.json", manifest(cfg, crawl, layout, fetched, published))
    (folder / "card.md").write_text(card(metrics), encoding="utf-8", newline="\n")
    return metrics


def manifest(cfg, crawl, layout, fetched, published):
    ledger = sorted((layout.root / "ledger" / crawl.id).glob("*.parquet"))
    return {
        "version": __version__,
        "crawl": crawl.id,
        "config": {k: v for k, v in cfg.raw.items() if k not in ("run", "fetch")},
        "seed": cfg.run.seed,
        "inputs": [{"index": i, "path": m["path"], "bytes": m["bytes"], "sha256": m["sha256"], "pages": m["pages"]}
                   for i, m in sorted(fetched.items())],
        "corpus": [{"file": f"corpus/{crawl.id}/{i:05d}.parquet", "docs": m["docs"], "bytes": m["bytes"],
                    "sha256": m["sha256"]} for i, m in sorted(published.items())],
        "ledger": [{"file": f"ledger/{crawl.id}/{p.name}", "sha256": sha256(p)} for p in ledger],
    }


def _pct(part, whole):
    return f"{100.0 * part / whole:.2f}%" if whole else "0.00%"


def card(m):
    pages = m["input"]["pages"]
    out = [f"# Run card: {m['crawl']}", ""]
    out.append(f"{m['shards']} WET files, {pages:,} pages, {m['input']['gz_bytes'] / 2**30:.2f} GiB compressed.")
    out.append(f"Published {m['published']['docs']:,} documents ({_pct(m['published']['docs'], pages)} of pages), "
               f"{m['published']['words']:,} words.")
    out += ["", "## Funnel", "", "| Step | Dropped | Remaining | Remaining share |", "|---|---:|---:|---:|"]
    for row in m["funnel"]:
        out.append(f"| {row['step']} | {row['dropped']:,} | {row['remaining']:,} | {_pct(row['remaining'], pages)} |")
    out += ["", "## Top drop reasons", "", "| Reason | Pages |", "|---|---:|"]
    for reason, count in list(m["reasons"].items())[:15]:
        out.append(f"| {reason} | {count:,} |")
    d = m["dedup"]
    c = d["cluster"]
    lines = m.get("c4_lines") or {}
    if lines:
        parts = ", ".join(f"{name.removeprefix('c4_').removesuffix('_line').replace('_', ' ')} {count:,}"
                          for name, count in sorted(lines.items(), key=lambda kv: -kv[1]))
        out += ["", f"C4 line rules removed {sum(lines.values()):,} lines from pages that passed the filters "
                f"({parts})."]
    out += ["", "## Dedup", "",
            f"Line dedup removed {d['lines_removed']:,} of {d['lines_seen']:,} lines "
            f"({_pct(d['lines_removed'], d['lines_seen'])}) from documents that stayed.",
            f"MinHash found {c.get('clusters', 0):,} near-duplicate clusters, the largest with "
            f"{c.get('largest_cluster', 0):,} documents.",
            f"{c.get('prior_overlap', 0):,} documents matched an earlier crawl through LSH, "
            f"{c.get('url_dup_prior', 0):,} URLs and {c.get('exact_dup_prior', 0):,} exact texts were "
            "already seen in earlier crawls."]
    e = m.get("eval") or {}
    lsh = e.get("lsh")
    if lsh:
        out += ["", "## Near-duplicate check against an exact similarity join", "",
                f"{lsh['sample_docs']:,} documents from {lsh['sample_shards']} shards, "
                f"{lsh['pairs_above_jaccard']:,} pairs at Jaccard >= {lsh['jaccard']}, "
                f"{lsh['docs_with_earlier_duplicate']:,} documents with an earlier near duplicate.", "",
                "| Min similarity | Removed | Within 0.75 of kept doc | Within 0.5 | Doc recall | Pair recall |",
                "|---:|---:|---:|---:|---:|---:|"]
        for row in lsh["variants"]:
            out.append(f"| {row['min_similarity']:.2f} | {row['removed']:,} | {row['removed_close']:.4f} | "
                       f"{row['removed_loose']:.4f} | {row['doc_recall']:.4f} | {row['pair_recall']:.4f} |")
        out += ["", "| Jaccard bin | Pairs | Measured candidate rate | Theory |", "|---|---:|---:|---:|"]
        for row in lsh["curve"]:
            out.append(f"| {row['low']:.2f} to {row['high']:.2f} | {row['pairs']:,} | {row['measured']:.4f} | "
                       f"{row['theory']:.4f} |")
    bloom = e.get("bloom")
    if bloom:
        out += ["", "## Bloom filter", "",
                f"{bloom['bits'] / 8 / 2**20:.1f} MiB, {bloom['hashes']} hashes, fill {bloom['fill']:.4f}.",
                f"Expected false positive rate {bloom['expected_fpr']:.2e}, probe rate {bloom['probe_fpr']:.2e} "
                f"over {bloom['probes']:,} random keys."]
        if "exact_fpr" in bloom:
            out.append(f"Against exact hashing: {bloom['exact_false_positives']:,} false positives in "
                       f"{bloom['exact_new']:,} new lines ({bloom['exact_fpr']:.2e}), "
                       f"{bloom['exact_false_negatives']} false negatives.")
    lang = m["language"]
    out += ["", "## Language ID", "",
            f"fastText and Common Crawl's own tag agree on {_pct(lang['agreement'], 1)} of "
            f"{lang['with_cc_header']:,} tagged pages.",
            f"{_pct(lang['english_overlap_of_fasttext'], 1)} of pages fastText kept as English are also tagged "
            f"English by Common Crawl."]
    timing = m["seconds"]
    out += ["", "## Cost", "",
            f"{timing['total'] / 60:.1f} minutes with {m['workers']} workers, "
            f"{m['pages_per_second']:.0f} pages per second.",
            f"Peak memory {m['peak_mb']['worker']:.0f} MB per worker, {m['peak_mb']['main']:.0f} MB main process.",
            f"PII redacted: {m['pii']['emails']:,} emails, {m['pii']['ips']:,} IPs, {m['pii']['phones']:,} phones.",
            ""]
    return "\n".join(out)
