import argparse
import sys
from pathlib import Path

from refinery import __version__, evaluate, report
from refinery.config import ConfigError, load
from refinery.fetch import FetchError
from refinery.langid import LangId
from refinery.pipeline import banned_words, check_state, finish_crawl, markers, process_crawl
from refinery.rules import Rules
from refinery.runner import Clock, State, WorkerCrash, read_json, write_json
from refinery.store import Layout
from refinery.wet import WetError


def log(message):
    print(message, flush=True)


def run(cfg):
    layout = Layout(cfg.run.data_dir)
    state = State(layout.state())
    check_state(cfg, state)
    banned = banned_words(cfg)
    for crawl in cfg.crawls:
        info = state.crawl(crawl.id)
        if info and info.get("done"):
            log(f"{crawl.id} is already done")
            continue
        log(f"{crawl.id}: {crawl.shards} WET files with {cfg.run.workers} workers")
        shards = process_crawl(cfg, crawl, layout, state, banned, log)
        info = state.crawl(crawl.id)
        if "eval" not in info:
            clock = Clock()
            result = {}
            if cfg.eval.shards:
                log(f"eval     {crawl.id}  exact similarity join on {min(cfg.eval.shards, len(shards))} shards")
                result = {"lsh": evaluate.lsh_quality(cfg, crawl, layout, shards),
                          "bloom": evaluate.bloom_quality(cfg, crawl, layout, state, shards)}
            write_json(layout.report(crawl.id) / "eval.json", result)
            info["eval"] = True
            info["seconds"]["eval"] = clock.seconds()
            state.commit(crawls={crawl.id: info})
        metrics = report.build(cfg, crawl, layout, state, shards, read_json(layout.report(crawl.id) / "eval.json"))
        finish_crawl(cfg, crawl, layout, state)
        log(f"{crawl.id} done: {metrics['published']['docs']:,} of {metrics['input']['pages']:,} pages published, "
            f"card at {layout.report(crawl.id) / 'card.md'}")
    return 0


def status(cfg):
    layout = Layout(cfg.run.data_dir)
    state = State(layout.state())
    for crawl in cfg.crawls:
        info = state.crawl(crawl.id)
        if info is None:
            log(f"{crawl.id}: not started")
            continue
        if info.get("done"):
            log(f"{crawl.id}: done")
            continue
        parts = [f"{stage} {len(markers(layout, crawl.id, stage))}/{crawl.shards}"
                 for stage in ("filter", "minhash", "publish")]
        parts.insert(2, "cluster done" if "cluster" in info else "cluster pending")
        parts.insert(3, f"lines {info.get('lines_next', 0)}/{crawl.shards}")
        log(f"{crawl.id}: " + ", ".join(parts))
    return 0


def explain(cfg, path, url):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    verdict = Rules(cfg, banned_words(cfg), LangId()).judge(url, text, explain=True)
    for check in verdict.checks:
        mark = "pass" if check.ok else "FAIL"
        log(f"{mark}  {check.name:<26} {check.value:>12.4f}  limit {check.limit:g}")
    for index, why in verdict.removed:
        log(f"line {index + 1} removed: {why}")
    log(f"verdict: {'kept' if not verdict.reason else 'dropped by ' + verdict.reason}  "
        f"(language {verdict.lang} {verdict.score:.3f})")
    return 0


def export(cfg, out):
    layout = Layout(cfg.run.data_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    crawls = []
    for crawl in cfg.crawls:
        metrics = layout.report(crawl.id) / "metrics.json"
        if metrics.exists():
            write_json(out / f"metrics-{crawl.id}.json", read_json(metrics))
            crawls.append(crawl.id)
    if not crawls:
        log("no finished crawls yet; run the pipeline first")
        return 1
    write_json(out / "crawls.json", crawls)
    log(f"wrote metrics for {len(crawls)} crawls to {out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="refinery", description="Common Crawl to pretraining corpus refinery")
    parser.add_argument("--version", action="version", version=f"refinery {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "explain", "export"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="config.toml")
        if name == "run":
            cmd.add_argument("--workers", type=int)
        if name == "explain":
            cmd.add_argument("file")
            cmd.add_argument("--url", default="https://example.com/page")
        if name == "export":
            cmd.add_argument("--out", default="space")
    args = parser.parse_args(argv)
    try:
        cfg = load(args.config)
        if args.command == "run":
            if args.workers:
                cfg.run.workers = max(1, args.workers)
            return run(cfg)
        if args.command == "status":
            return status(cfg)
        if args.command == "explain":
            return explain(cfg, args.file, args.url)
        return export(cfg, args.out)
    except (ConfigError, FetchError, WetError, WorkerCrash) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("stopped; run the same command again to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
