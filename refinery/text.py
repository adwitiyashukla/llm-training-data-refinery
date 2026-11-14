import re
import unicodedata
from urllib.parse import urlsplit

import xxhash

DIGITS = re.compile(r"\d+")
NON_WORD = re.compile(r"[\W_]+")
URL_SPLIT = re.compile(r"[^a-z0-9]+")


def hash64(text):
    return xxhash.xxh3_64_intdigest(text.encode("utf-8", "surrogatepass"))


def fold(text):
    text = text.lower()
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def simplify(text):
    text = DIGITS.sub("0", fold(text))
    return NON_WORD.sub(" ", text).strip()


def line_hashes(text):
    out = []
    for line in text.split("\n"):
        key = simplify(line)
        out.append(hash64(key) if key else 0)
    return out


def text_hash(text):
    return hash64(simplify(text))


def _parts(url):
    try:
        parts = urlsplit(url.strip())
        return (parts.hostname or "").lower(), parts.path, parts.query
    except ValueError:
        return "", url.strip(), ""


def url_key(url):
    host, path, query = _parts(url)
    if host.startswith("www."):
        host = host[4:]
    path = path.rstrip("/") or "/"
    return f"{host}{path}?{query}" if query else f"{host}{path}"


def url_tokens(url):
    host, path, query = _parts(url)
    blob = f"{host} {path} {query}".lower()
    return {t for t in URL_SPLIT.split(blob) if t}


def count_words(text):
    return len(text.split())
