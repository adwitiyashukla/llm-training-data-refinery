import gzip
import uuid
from typing import NamedTuple


class WetError(ValueError):
    pass


class Page(NamedTuple):
    url: str
    date: str
    record_id: str
    lang: str
    text: str


def records(stream):
    while True:
        line = stream.readline()
        if not line:
            return
        if not line.strip():
            continue
        if not line.startswith(b"WARC/"):
            raise WetError(f"expected a WARC version line, got {line[:40]!r}")
        headers = {}
        while True:
            line = stream.readline()
            if not line:
                raise WetError("file ended inside a record header")
            line = line.rstrip(b"\r\n")
            if not line:
                break
            key, sep, value = line.partition(b":")
            if not sep:
                raise WetError(f"bad header line {line[:40]!r}")
            headers[key.strip().lower().decode("ascii", "replace")] = value.strip().decode("utf-8", "replace")
        try:
            size = int(headers["content-length"])
        except (KeyError, ValueError):
            raise WetError("record without a valid Content-Length") from None
        body = stream.read(size)
        if len(body) != size:
            raise WetError("file ended inside a record body")
        yield headers, body


def pages(path):
    try:
        yield from _pages(path)
    except (EOFError, gzip.BadGzipFile, OSError) as e:
        raise WetError(f"{path} is damaged: {e}") from None


def _pages(path):
    with gzip.open(path, "rb") as stream:
        for headers, body in records(stream):
            if headers.get("warc-type") != "conversion":
                continue
            text = body.decode("utf-8", "replace")
            if "\r" in text:
                text = text.replace("\r\n", "\n").replace("\r", "\n")
            yield Page(
                headers.get("warc-target-uri", ""),
                headers.get("warc-date", ""),
                headers.get("warc-record-id", ""),
                headers.get("warc-identified-content-language", ""),
                text.strip(),
            )


def _record(headers, body):
    head = ["WARC/1.0"] + [f"{k}: {v}" for k, v in headers] + [f"Content-Length: {len(body)}"]
    return ("\r\n".join(head) + "\r\n\r\n").encode() + body + b"\r\n\r\n"


def write(path, items, date="2026-06-15T10:00:00Z"):
    info = b"software: llm-training-data-refinery\r\nformat: WARC File Format 1.0\r\n"
    with open(path, "wb") as out:
        out.write(gzip.compress(_record([("WARC-Type", "warcinfo"), ("WARC-Date", date)], info), mtime=0))
        for page in items:
            url, text = page[0], page[1]
            lang = page[2] if len(page) > 2 else ""
            rid = uuid.uuid5(uuid.NAMESPACE_URL, url)
            headers = [
                ("WARC-Type", "conversion"),
                ("WARC-Target-URI", url),
                ("WARC-Date", date),
                ("WARC-Record-ID", f"<urn:uuid:{rid}>"),
            ]
            if lang:
                headers.append(("WARC-Identified-Content-Language", lang))
            headers.append(("Content-Type", "text/plain"))
            out.write(gzip.compress(_record(headers, text.encode("utf-8")), mtime=0))
