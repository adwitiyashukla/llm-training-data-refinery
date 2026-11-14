import random
from collections import Counter

import pytest
from conftest import english, foreign, settings

from refinery.langid import LangId
from refinery.rules import Rules, _ngram_chars, c4, fineweb, quality, repetition
from refinery.text import hash64


@pytest.fixture(scope="module")
def lid():
    return LangId()


@pytest.fixture
def cfg(tmp_path):
    return settings(tmp_path, tmp_path, [("CC-MAIN-2099-01", 1)])


def failed(checks):
    return [c.name for c in checks if not c.ok]


def test_clean_english_passes_every_check(cfg, lid):
    text = english(random.Random(1), 10)
    verdict = Rules(cfg, frozenset(), lid).judge("https://a.example/news", text, explain=True)
    assert verdict.reason == ""
    assert verdict.lang == "en" and verdict.score > 0.9
    assert all(c.ok for c in verdict.checks)
    assert len(verdict.checks) == 2 + 13 + 10 + 3 + 3


def test_foreign_text_fails_language(cfg, lid):
    verdict = Rules(cfg, frozenset(), lid).judge("https://b.example/", foreign(random.Random(2), "de"))
    assert verdict.reason == "lang_score"
    assert verdict.lang == "de"


def test_blocklist_matches_whole_url_tokens(cfg, lid):
    rules = Rules(cfg, frozenset([hash64("blockedtoken")]), lid)
    text = english(random.Random(3))
    assert rules.judge("https://blockedtoken.example/a", text).reason == "url_blocklist"
    assert rules.judge("https://x.example/blockedtoken-deals", text).reason == "url_blocklist"
    assert rules.judge("https://notblockedtokens.example/a", text).reason == ""


def test_repeated_lines(cfg):
    text = "\n".join(["Buy cheap watches online today best price free shipping"] * 10)
    assert "rep_dup_lines" in failed(repetition(text, cfg.gopher_repetition))


def test_repeated_ngrams(cfg):
    base = english(random.Random(4), 3)
    text = base + "\n" + base.replace("\n", " ")
    names = failed(repetition(text, cfg.gopher_repetition))
    assert any(name.startswith("rep_dup_") and name.endswith("gram") for name in names)


def brute_ngrams(words):
    top, dup = {}, {}
    for n in (2, 3, 4):
        grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        if grams:
            gram, count = grams.most_common(1)[0]
            top[n] = len(" ".join(gram)) * count
    for n in range(5, 11):
        seen, covered = set(), set()
        for i in range(len(words) - n + 1):
            gram = tuple(words[i : i + n])
            if gram in seen:
                covered.update(range(i, i + n))
            seen.add(gram)
        if covered:
            dup[n] = sum(len(words[i]) for i in covered)
    return top, dup


def test_ngram_stats_match_brute_force():
    rng = random.Random(5)
    vocab = ["a", "bb", "ccc", "dddd", "e", "ff"]
    for _ in range(200):
        words = [rng.choice(vocab) for _ in range(rng.randint(0, 60))]
        assert _ngram_chars(words) == brute_ngrams(words)


def test_quality_rules(cfg):
    q = cfg.gopher_quality
    assert "gopher_few_words" in failed(quality("Too short to keep.", q))
    hashes = " ".join(f"#tag{i} word" for i in range(60))
    assert "gopher_hashes" in failed(quality(hashes, q))
    bullets = "\n".join(f"- item number {i} in the list" for i in range(30))
    assert "gopher_bullet_lines" in failed(quality(bullets, q))
    no_stops = " ".join(["quick brown foxes jump over lazy dogs"] * 10)
    assert "gopher_stop_words" in failed(quality(no_stops, q))
    numbers = " ".join(str(i) for i in range(300)) + " the and"
    assert "gopher_alpha_words" in failed(quality(numbers, q))


def test_c4_line_and_document_rules(cfg):
    c = cfg.c4
    body = english(random.Random(6), 4)
    checks, text, removed = c4(body + "\nPlease enable JavaScript to view this page.\nRead our privacy policy here.", c)
    assert not failed(checks)
    assert "JavaScript" not in text and "privacy policy" not in text
    assert {why for _, why in removed} == {"c4_javascript_line", "c4_policy_line"}
    checks, _, _ = c4(body + "\nvar x = { a: 1 };", c)
    assert "c4_curly_bracket" in failed(checks)
    checks, _, _ = c4(body + "\nLorem ipsum dolor sit amet.", c)
    assert "c4_lorem_ipsum" in failed(checks)
    checks, _, _ = c4("One short sentence here.\nAnother one there.", c)
    assert "c4_few_sentences" in failed(checks)
    _, text, removed = c4("The river rose again[1] after the storm.[citation needed]\nHi\n" + body, c)
    assert text.startswith("The river rose again after the storm.")
    assert (1, "c4_short_line") in removed


def test_c4_terminal_punctuation_switch(cfg):
    c = cfg.c4
    text = english(random.Random(7), 4) + "\nthis line has no ending mark"
    assert "this line has no ending mark" in c4(text, c)[1]
    c.terminal_punct = True
    assert "this line has no ending mark" not in c4(text, c)[1]


def test_fineweb_rules(cfg):
    f = cfg.fineweb
    no_punct = "\n".join(f"heading number {i} without any end mark here" for i in range(20))
    assert "fineweb_punct_lines" in failed(fineweb(no_punct, f))
    short = "\n".join(f"Item {i} is listed." for i in range(20))
    assert "fineweb_short_lines" in failed(fineweb(short, f))
    line = "This exact sentence shows up again and again in the same document."
    dup = "\n".join([line] * 3 + [english(random.Random(8), 1)])
    assert "fineweb_dup_line_chars" in failed(fineweb(dup, f))


def test_first_failure_is_reported_in_pipeline_order(cfg, lid):
    rules = Rules(cfg, frozenset([hash64("blockedtoken")]), lid)
    verdict = rules.judge("https://blockedtoken.example/", "Bonjour tout le monde.", explain=True)
    assert verdict.reason == "url_blocklist"
    assert "lang_score" in failed(verdict.checks)
