import re
from typing import NamedTuple

import numpy as np

from refinery.text import hash64, url_tokens

STOP_WORDS = frozenset(["the", "be", "to", "of", "and", "that", "have", "with"])
POLICY = ("terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies", "use cookies")
BULLETS = ("\u2022", "-")
ELLIPSES = ("...", "\u2026")
C4_END = (".", "?", "!", '"', "'")
TERMINAL = (".", "!", "?", "\u3002", "\uff01", "\uff1f", "\u0964")
CITATION = re.compile(r"\[\d*\]|\[edit\]|\[citation needed\]")
PARAGRAPHS = re.compile(r"\n{2,}")
LINES = re.compile(r"\n+")
WORD = re.compile(r"\.\.\.|[^\W_]+(?:[.,'\u2019\-/:@%][^\W_]+)*|[^\w\s]|_+")
HAS_ALPHA = re.compile(r"[^\W\d_]")
SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")
STOP_FINDER = re.compile(r"(?<![^\W_])(?:" + "|".join(sorted(STOP_WORDS)) + r")(?![^\W_])")
MULT = np.uint64(0x9E3779B97F4A7C15)


class Check(NamedTuple):
    name: str
    value: float
    limit: float
    ok: bool


class Verdict(NamedTuple):
    reason: str
    value: float
    text: str
    lang: str
    score: float
    checks: list
    removed: list


def at_most(name, value, limit):
    return Check(name, float(value), float(limit), value <= limit)


def at_least(name, value, limit):
    return Check(name, float(value), float(limit), value >= limit)


def _dups(items):
    seen = set()
    count = chars = 0
    for item in items:
        if item in seen:
            count += 1
            chars += len(item)
        else:
            seen.add(item)
    return count, chars


def _ngram_chars(words):
    size = len(words)
    top, dup = {}, {}
    if size < 2:
        return top, dup
    ids = np.fromiter(map(hash, words), np.int64, size).view(np.uint64)
    lens = np.fromiter(map(len, words), np.int64, size)
    offsets = np.concatenate(([0], np.cumsum(lens)))
    keys = ids
    for n in range(2, 11):
        if size < n:
            break
        keys = keys[:-1] * MULT + ids[n - 1 :]
        _, first, inverse, counts = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
        if n <= 4:
            best = counts.max()
            start = first[counts == best].min()
            top[n] = int(offsets[start + n] - offsets[start] + n - 1) * int(best)
            continue
        repeats = np.flatnonzero(first[inverse] != np.arange(keys.size))
        if repeats.size:
            cover = np.zeros(size + 1, np.int64)
            np.add.at(cover, repeats, 1)
            np.add.at(cover, repeats + n, -1)
            dup[n] = int(lens[np.cumsum(cover[:size]) > 0].sum())
    return top, dup


def repetition(text, c):
    length = max(1, len(text))
    paragraphs = PARAGRAPHS.split(text.strip())
    para_dups, para_chars = _dups(paragraphs)
    lines = LINES.split(text)
    line_dups, line_chars = _dups(lines)
    checks = [
        at_most("rep_dup_paragraphs", para_dups / len(paragraphs), c.dup_para_frac),
        at_most("rep_dup_paragraph_chars", para_chars / length, c.dup_para_char_frac),
        at_most("rep_dup_lines", line_dups / len(lines), c.dup_line_frac),
        at_most("rep_dup_line_chars", line_chars / length, c.dup_line_char_frac),
    ]
    top, dup = _ngram_chars(WORD.findall(text))
    for n, limit in zip((2, 3, 4), c.top_ngram_char_frac, strict=True):
        checks.append(at_most(f"rep_top_{n}gram", top.get(n, 0) / length, limit))
    for n, limit in zip(range(5, 11), c.dup_ngram_char_frac, strict=True):
        checks.append(at_most(f"rep_dup_{n}gram", dup.get(n, 0) / length, limit))
    return checks


def quality(text, c):
    tokens = WORD.findall(text)
    total = max(1, len(tokens))
    words = [t for t in tokens if t[0].isalnum()]
    mean = sum(map(len, words)) / len(words) if words else 0.0
    lines = text.splitlines() or [""]
    bullets = sum(1 for line in lines if line.lstrip().startswith(BULLETS))
    trailing = sum(1 for line in lines if line.rstrip().endswith(ELLIPSES))
    alpha = sum(1 for t in tokens if HAS_ALPHA.search(t))
    stops = len(set(STOP_FINDER.findall(text.lower())))
    return [
        at_least("gopher_few_words", len(words), c.min_words),
        at_most("gopher_many_words", len(words), c.max_words),
        at_least("gopher_short_words", mean, c.min_mean_word_len),
        at_most("gopher_long_words", mean, c.max_mean_word_len),
        at_most("gopher_hashes", text.count("#") / total, c.max_hash_ratio),
        at_most("gopher_ellipses", (text.count("...") + text.count("\u2026")) / total, c.max_ellipsis_ratio),
        at_most("gopher_bullet_lines", bullets / len(lines), c.max_bullet_lines),
        at_most("gopher_ellipsis_lines", trailing / len(lines), c.max_ellipsis_lines),
        at_least("gopher_alpha_words", alpha / total, c.min_alpha_words),
        at_least("gopher_stop_words", stops, c.min_stop_words),
    ]


def sentences(line):
    found = len(SENTENCE_END.findall(line))
    return found if line.rstrip("\"')]").endswith((".", "!", "?")) else found + 1


def c4(text, c):
    kept, removed, count = [], [], 0
    lorem = curly = False
    for index, raw in enumerate(text.split("\n")):
        line = raw.strip()
        if not line:
            continue
        words = line.split()
        why = ""
        if len(line) > c.max_word_len and any(len(w) > c.max_word_len for w in words):
            why = "c4_long_word_line"
        else:
            if c.strip_citations:
                line = CITATION.sub("", line).strip()
            low = line.lower()
            if c.terminal_punct and (not line.endswith(C4_END) or line.endswith("...")):
                why = "c4_no_punct_line"
            elif len(words) < c.min_words_per_line:
                why = "c4_short_line"
            elif c.drop_lorem_ipsum and "lorem ipsum" in low:
                lorem = True
                why = "c4_lorem_ipsum"
            elif c.drop_javascript_lines and "javascript" in low:
                why = "c4_javascript_line"
            elif c.drop_curly_bracket and "{" in line:
                curly = True
                why = "c4_curly_bracket"
            elif c.drop_policy_lines and any(p in low for p in POLICY):
                why = "c4_policy_line"
        if why:
            removed.append((index, why))
            continue
        count += sentences(line)
        kept.append(line)
    checks = [
        Check("c4_lorem_ipsum", float(lorem), 0.0, not lorem),
        Check("c4_curly_bracket", float(curly), 0.0, not curly),
        at_least("c4_few_sentences", count, c.min_sentences),
    ]
    return checks, "\n".join(kept), removed


def fineweb(text, c):
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return [Check("fineweb_punct_lines", 0.0, c.min_punct_lines, False)]
    punct = sum(1 for line in lines if line.rstrip().endswith(TERMINAL)) / len(lines)
    short = sum(1 for line in lines if len(line) <= c.short_line_chars) / len(lines)
    dup_chars = _dups(lines)[1] / max(1, len(text) - text.count("\n"))
    return [
        at_least("fineweb_punct_lines", punct, c.min_punct_lines),
        at_most("fineweb_short_lines", short, c.max_short_lines),
        at_most("fineweb_dup_line_chars", dup_chars, c.max_dup_line_chars),
    ]


class Rules:
    def __init__(self, cfg, banned=frozenset(), langid=None):
        self.cfg = cfg
        self.banned = frozenset(banned)
        self.langid = langid

    def judge(self, url, text, explain=False):
        cfg = self.cfg
        lang, score = self.langid(text) if self.langid else ("", 0.0)
        hits = sum(1 for token in url_tokens(url) if hash64(token) in self.banned) if self.banned else 0
        groups = [
            [Check("url_blocklist", float(hits), 0.0, hits == 0)],
            [Check("lang_score", score, cfg.language.min_score,
                   lang in cfg.language.keep and score >= cfg.language.min_score)],
        ]
        if not explain:
            for checks in groups:
                for check in checks:
                    if not check.ok:
                        return Verdict(check.name, check.value, text, lang, score, [], [])
        found = []
        for step in ("repetition", "quality", "c4", "fineweb"):
            removed = []
            if step == "repetition":
                checks = repetition(text, cfg.gopher_repetition)
            elif step == "quality":
                checks = quality(text, cfg.gopher_quality)
            elif step == "c4":
                checks, text, removed = c4(text, cfg.c4)
            else:
                checks = fineweb(text, cfg.fineweb)
            groups.append(checks)
            found.extend(removed)
            if not explain:
                for check in checks:
                    if not check.ok:
                        return Verdict(check.name, check.value, text, lang, score, [], [])
        flat = [check for checks in groups for check in checks]
        first = next((check for check in flat if not check.ok), None)
        if first is None:
            return Verdict("", 0.0, text, lang, score, flat if explain else [], found)
        return Verdict(first.name, first.value, text, lang, score, flat, found)


def parse_blocklist(data):
    out = set()
    for line in data.decode("ascii", "replace").split():
        if len(line) != 16:
            raise ValueError(f"bad blocklist line {line[:20]!r}")
        out.add(int(line, 16))
    return frozenset(out)
