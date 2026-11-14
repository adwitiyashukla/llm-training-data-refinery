from refinery.pii import EMAIL_SUB, IP_SUB, PHONE_SUB, redact
from refinery.text import line_hashes, simplify, text_hash, url_key, url_tokens


def test_simplify():
    assert simplify("Caf\u00e9 OPENS at 9:30, 2026!") == "cafe opens at 0 0 0"
    assert simplify("  multiple   spaces\tand_tabs ") == "multiple spaces and tabs"


def test_text_hash_ignores_case_digits_and_punctuation():
    assert text_hash("Hello, World 2025") == text_hash("hello world 1999!")
    assert text_hash("hello world") != text_hash("hello there")


def test_line_hashes_keep_empty_lines_as_zero():
    hashes = line_hashes("First line.\n\n---\nFirst line!")
    assert hashes[0] == hashes[3] != 0
    assert hashes[1] == 0 and hashes[2] == 0


def test_url_key_normalizes():
    assert url_key("https://WWW.Example.com/path/") == url_key("http://example.com/path")
    assert url_key("https://example.com/a?b=1") != url_key("https://example.com/a?b=2")
    assert url_key("https://example.com/a#frag") == url_key("https://example.com/a")
    assert url_key("http://[::1") == "http://[::1"


def test_url_tokens():
    assert url_tokens("https://shop.example.com/best-deals/2026?q=red+shoes") >= {"shop", "best", "deals", "shoes"}
    assert url_tokens("https://WWW.Example.org:8080/x") == {"www", "example", "org", "x"}


def test_redact_emails_ips_phones():
    text = ("Mail jane.doe+news@mail.example.co.uk or bob@example.org. Server 8.8.8.8 is public, "
            "192.168.1.20 is private. Call (513) 555-0199, 513.555.0123 or +44 20 7946 0958 today.")
    out, counts = redact(text)
    assert counts == {"emails": 2, "ips": 1, "phones": 3}
    assert "jane.doe" not in out and "8.8.8.8" not in out and "7946" not in out
    assert "192.168.1.20" in out
    assert out.count(EMAIL_SUB) == 2 and out.count(IP_SUB) == 1 and out.count(PHONE_SUB) == 3


def test_redact_leaves_numbers_alone():
    text = "Version 1.2.3 shipped on 2026-10-06 for $1,234.56 with id 5135550199 and 1.2.3.4.5 build."
    out, counts = redact(text)
    assert out == text
    assert counts == {"emails": 0, "ips": 0, "phones": 0}


def test_redact_switches():
    text = "Write to a@example.org from 8.8.4.4 or call 513-555-0100."
    out, counts = redact(text, emails=False, ips=False, phones=False)
    assert out == text and sum(counts.values()) == 0
