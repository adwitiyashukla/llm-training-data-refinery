import gzip
import os
import random
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from refinery import __version__

AGENT = f"llm-training-data-refinery/{__version__} (+https://github.com/adwitiyashukla/llm-training-data-refinery)"
RETRY_CODES = {408, 425, 429, 500, 502, 503, 504}
TOTAL = re.compile(r"/(\d+)$")


class FetchError(RuntimeError):
    pass


def _expected(resp, have):
    span = resp.headers.get("Content-Range")
    if span:
        found = TOTAL.search(span)
        return int(found.group(1)) if found else None
    size = resp.headers.get("Content-Length")
    return have + int(size) if size else None


def download(url, dest, timeout=60.0, retries=6, sleep=time.sleep):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": AGENT}
        if have and url.startswith("http"):
            headers["Range"] = f"bytes={have}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
                status = getattr(resp, "status", None) or 200
                if have and status != 206:
                    have = 0
                total = _expected(resp, have)
                with open(part, "ab" if have else "wb") as out:
                    while block := resp.read(1 << 20):
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
            size = part.stat().st_size
            if total is not None and size != total:
                raise FetchError(f"got {size} of {total} bytes")
            os.replace(part, dest)
            return dest
        except HTTPError as e:
            last = e
            if e.code == 416:
                part.unlink(missing_ok=True)
            elif e.code not in RETRY_CODES:
                raise FetchError(f"{url}: HTTP {e.code}") from None
        except (URLError, TimeoutError, ConnectionError, OSError, FetchError) as e:
            last = e
        if attempt < retries:
            sleep(min(60.0, 2.0**attempt) * (0.5 + random.random()))
    raise FetchError(f"{url}: gave up after {retries + 1} tries ({last})")


def shard_list(base_url, crawl_id, folder, timeout=60.0, retries=6):
    path = Path(folder) / "wet.paths.gz"
    if not path.exists():
        download(f"{base_url}crawl-data/{crawl_id}/wet.paths.gz", path, timeout, retries)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def pick(paths, count):
    count = min(count, len(paths))
    return [(k * len(paths) // count, paths[k * len(paths) // count]) for k in range(count)]
