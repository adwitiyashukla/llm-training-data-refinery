import hashlib
import importlib.resources
import json
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

CRAWL_ID = re.compile(r"^CC-MAIN-\d{4}-\d{2}$")

SCHEMA = {
    "run": {
        "data_dir": ("str", "data"),
        "workers": ("int", 1, 64, 4),
        "seed": ("int", 0, 2**31 - 1, 1234),
        "keep_raw": ("bool", False),
        "keep_work": ("bool", False),
        "batch_docs": ("int", 100, 50000, 2000),
    },
    "fetch": {
        "base_url": ("str", "https://data.commoncrawl.org/"),
        "timeout": ("float", 1.0, 600.0, 60.0),
        "retries": ("int", 0, 20, 6),
    },
    "url": {
        "enabled": ("bool", True),
        "blocklist": ("str", ""),
    },
    "language": {
        "keep": ("strs", ["en"]),
        "min_score": ("float", 0.0, 1.0, 0.65),
    },
    "gopher_repetition": {
        "dup_para_frac": ("float", 0.0, 1.0, 0.30),
        "dup_para_char_frac": ("float", 0.0, 1.0, 0.20),
        "dup_line_frac": ("float", 0.0, 1.0, 0.30),
        "dup_line_char_frac": ("float", 0.0, 1.0, 0.20),
        "top_ngram_char_frac": ("floats", 3, 0.0, 1.0, [0.20, 0.18, 0.16]),
        "dup_ngram_char_frac": ("floats", 6, 0.0, 1.0, [0.15, 0.14, 0.13, 0.12, 0.11, 0.10]),
    },
    "gopher_quality": {
        "min_words": ("int", 0, 10**7, 50),
        "max_words": ("int", 1, 10**7, 100000),
        "min_mean_word_len": ("float", 0.0, 100.0, 3.0),
        "max_mean_word_len": ("float", 0.0, 100.0, 10.0),
        "max_hash_ratio": ("float", 0.0, 10.0, 0.1),
        "max_ellipsis_ratio": ("float", 0.0, 10.0, 0.1),
        "max_bullet_lines": ("float", 0.0, 1.0, 0.9),
        "max_ellipsis_lines": ("float", 0.0, 1.0, 0.3),
        "min_alpha_words": ("float", 0.0, 1.0, 0.8),
        "min_stop_words": ("int", 0, 8, 2),
    },
    "c4": {
        "min_words_per_line": ("int", 0, 1000, 3),
        "min_sentences": ("int", 0, 1000, 5),
        "max_word_len": ("int", 1, 10**6, 1000),
        "terminal_punct": ("bool", False),
        "strip_citations": ("bool", True),
        "drop_lorem_ipsum": ("bool", True),
        "drop_curly_bracket": ("bool", True),
        "drop_javascript_lines": ("bool", True),
        "drop_policy_lines": ("bool", True),
    },
    "fineweb": {
        "min_punct_lines": ("float", 0.0, 1.0, 0.12),
        "max_short_lines": ("float", 0.0, 1.0, 0.67),
        "short_line_chars": ("int", 1, 10000, 30),
        "max_dup_line_chars": ("float", 0.0, 1.0, 0.10),
    },
    "dedup": {
        "cross_crawl": ("bool", True),
        "min_words": ("int", 0, 10**6, 50),
        "bloom_capacity": ("int", 1000, 10**10, 20_000_000),
        "bloom_fpr": ("float", 1e-9, 0.5, 0.001),
        "shingle_words": ("int", 1, 50, 5),
        "bands": ("int", 1, 512, 14),
        "rows": ("int", 1, 64, 8),
        "min_similarity": ("float", 0.0, 1.0, 0.7),
    },
    "pii": {
        "emails": ("bool", True),
        "ips": ("bool", True),
        "phones": ("bool", True),
    },
    "eval": {
        "shards": ("int", 0, 100000, 20),
        "max_docs": ("int", 100, 10**7, 40000),
        "jaccard": ("float", 0.05, 1.0, 0.75),
        "floor": ("float", 0.05, 1.0, 0.5),
    },
}

class ConfigError(ValueError):
    pass


def _check(section, key, spec, value, errors):
    kind = spec[0]
    where = f"[{section}] {key}"
    if kind == "bool":
        if not isinstance(value, bool):
            errors.append(f"{where} must be true or false")
        return value
    if kind == "str":
        if not isinstance(value, str):
            errors.append(f"{where} must be a string")
        return value
    if kind == "strs":
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
            errors.append(f"{where} must be a non-empty list of strings")
        return value
    if kind == "floats":
        size, lo, hi = spec[1], spec[2], spec[3]
        ok = isinstance(value, list) and len(value) == size
        ok = ok and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        if not ok or not all(lo <= v <= hi for v in value):
            errors.append(f"{where} must be a list of {size} numbers between {lo} and {hi}")
            return value
        return [float(v) for v in value]
    lo, hi = spec[1], spec[2]
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{where} must be an integer")
            return value
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{where} must be a number")
        return value
    if not lo <= value <= hi:
        errors.append(f"{where} must be between {lo} and {hi}, got {value}")
    return float(value) if kind == "float" else value


def _crawls(raw, errors):
    if not isinstance(raw, list) or not raw:
        errors.append("config needs at least one [[crawls]] entry")
        return []
    out, seen = [], set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"[[crawls]] entry {i + 1} must be a table")
            continue
        extra = set(item) - {"id", "shards"}
        if extra:
            errors.append(f"[[crawls]] entry {i + 1} has unknown keys: {', '.join(sorted(extra))}")
        crawl_id, shards = item.get("id"), item.get("shards")
        if not isinstance(crawl_id, str) or not CRAWL_ID.match(crawl_id):
            errors.append(f"[[crawls]] entry {i + 1} needs an id like CC-MAIN-2026-25")
        elif crawl_id in seen:
            errors.append(f"[[crawls]] {crawl_id} is listed twice")
        seen.add(crawl_id)
        if isinstance(shards, bool) or not isinstance(shards, int) or not 1 <= shards <= 100000:
            errors.append(f"[[crawls]] entry {i + 1} needs shards between 1 and 100000")
        out.append(SimpleNamespace(id=crawl_id, shards=shards, number=i + 1))
    return out


def parse(raw, base_dir=None):
    errors = []
    unknown = set(raw) - set(SCHEMA) - {"crawls"}
    if unknown:
        errors.append(f"unknown sections: {', '.join(sorted(unknown))}")
    sections = {}
    for section, specs in SCHEMA.items():
        given = raw.get(section, {})
        if not isinstance(given, dict):
            errors.append(f"[{section}] must be a table")
            given = {}
        extra = set(given) - set(specs)
        if extra:
            errors.append(f"[{section}] has unknown keys: {', '.join(sorted(extra))}")
        values = {}
        for key, spec in specs.items():
            value = given.get(key, spec[-1])
            values[key] = _check(section, key, spec, value, errors)
        sections[section] = values
    crawls = _crawls(raw.get("crawls"), errors)
    gq = sections["gopher_quality"]
    if not errors and gq["min_mean_word_len"] > gq["max_mean_word_len"]:
        errors.append("[gopher_quality] min_mean_word_len is above max_mean_word_len")
    if not errors and gq["min_words"] > gq["max_words"]:
        errors.append("[gopher_quality] min_words is above max_words")
    if not errors and sections["eval"]["floor"] > sections["eval"]["jaccard"]:
        errors.append("[eval] floor must not be above jaccard")
    if errors:
        raise ConfigError("invalid config:\n  " + "\n  ".join(errors))
    data_dir = Path(sections["run"]["data_dir"])
    if base_dir is not None and not data_dir.is_absolute():
        data_dir = Path(base_dir) / data_dir
    sections["run"]["data_dir"] = str(data_dir)
    blocklist = sections["url"]["blocklist"]
    if blocklist and base_dir is not None and not Path(blocklist).is_absolute():
        sections["url"]["blocklist"] = str(Path(base_dir) / blocklist)
    cfg = SimpleNamespace(**{name: SimpleNamespace(**values) for name, values in sections.items()})
    cfg.crawls = crawls
    cfg.raw = sections
    return cfg


def load(path):
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "rb") as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"could not parse {path}: {e}") from None
    return parse(raw, base_dir=path.parent)


def blocklist_bytes(cfg):
    if not cfg.url.enabled:
        return b""
    if not cfg.url.blocklist:
        return importlib.resources.files("refinery").joinpath("blocklist.txt").read_bytes()
    path = Path(cfg.url.blocklist)
    if not path.is_file():
        raise ConfigError(f"[url] blocklist file not found: {path}")
    return path.read_bytes()


def fingerprint(cfg, crawl):
    shared = {k: v for k, v in cfg.raw.items() if k not in ("run", "fetch", "eval")}
    shared["seed"] = cfg.raw["run"]["seed"]
    shared["blocklist"] = hashlib.sha256(blocklist_bytes(cfg)).hexdigest()
    shared["crawl"] = [crawl.number, crawl.id, crawl.shards]
    blob = json.dumps(shared, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
