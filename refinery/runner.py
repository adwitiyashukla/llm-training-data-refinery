import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import psutil


class WorkerCrash(RuntimeError):
    pass


def crash_point(name):
    if os.environ.get("REFINERY_CRASH") == name:
        os._exit(3)


def sync(path):
    with open(path, "r+b") as f:
        os.fsync(f.fileno())


def promote(tmp, final):
    sync(tmp)
    os.replace(tmp, final)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_mb():
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is not None:
        return peak / 2**20
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 2**20 if sys.platform == "darwin" else usage / 1024


def clean_tmp(folder):
    folder = Path(folder)
    if folder.exists():
        for path in folder.rglob("*.tmp"):
            path.unlink()


def run_tasks(fn, jobs, workers, on_done):
    if not jobs:
        return
    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            on_done(job, fn(job))
        return
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context)
    try:
        pending = {pool.submit(fn, job): job for job in jobs}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception as e:
                    if type(e).__name__ == "BrokenProcessPool":
                        raise WorkerCrash("a worker process died; run the same command again to resume") from e
                    raise
                on_done(job, result)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()


class State:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.head = read_json(self.root / "head.json", {"version": 0, "files": {}, "crawls": {}})

    def path(self, name):
        file = self.head["files"].get(name)
        return self.root / file if file else None

    def fresh(self, name, suffix):
        return self.root / f"{name}.{self.head['version'] + 1}{suffix}"

    def crawl(self, crawl_id):
        return self.head["crawls"].get(crawl_id)

    def commit(self, files=None, crawls=None):
        head = json.loads(json.dumps(self.head))
        head["version"] += 1
        for name, path in (files or {}).items():
            head["files"][name] = Path(path).name
        for crawl_id, info in (crawls or {}).items():
            head["crawls"][crawl_id] = info
        write_json(self.root / "head.json", head)
        self.head = head
        keep = set(head["files"].values()) | {"head.json"}
        for path in self.root.iterdir():
            if path.is_file() and path.name not in keep:
                path.unlink()


class Clock:
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self):
        return time.perf_counter() - self.start
