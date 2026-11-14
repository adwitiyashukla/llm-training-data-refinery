import gzip

import pytest

from refinery import wet


def test_round_trip(tmp_path):
    path = tmp_path / "a.warc.wet.gz"
    items = [("https://a.example/1", "first line\nsecond line", "eng"), ("https://b.example/2", "hello", "")]
    wet.write(path, items)
    pages = list(wet.pages(path))
    assert [p.url for p in pages] == ["https://a.example/1", "https://b.example/2"]
    assert pages[0].text == "first line\nsecond line"
    assert pages[0].lang == "eng"
    assert pages[1].lang == ""
    assert all(p.record_id.startswith("<urn:uuid:") for p in pages)


def test_crlf_and_unicode(tmp_path):
    path = tmp_path / "b.warc.wet.gz"
    wet.write(path, [("https://c.example/", "caf\u00e9 menu\r\nline two\r\n", "fra")])
    page = next(wet.pages(path))
    assert page.text == "caf\u00e9 menu\nline two"


def test_reads_one_member_per_record(tmp_path):
    path = tmp_path / "c.warc.wet.gz"
    wet.write(path, [(f"https://d.example/{i}", f"text {i}") for i in range(50)])
    with gzip.open(path, "rb") as f:
        records = list(wet.records(f))
    assert len(records) == 51
    assert records[0][0]["warc-type"] == "warcinfo"


def test_truncated_file_raises(tmp_path):
    path = tmp_path / "d.warc.wet.gz"
    wet.write(path, [("https://e.example/", "x" * 5000)])
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 40])
    with pytest.raises(wet.WetError):
        list(wet.pages(path))


def test_bad_record_raises(tmp_path):
    path = tmp_path / "e.warc.wet.gz"
    path.write_bytes(gzip.compress(b"NOT A WARC\r\n\r\n"))
    with pytest.raises(wet.WetError):
        list(wet.pages(path))
    path.write_bytes(gzip.compress(b"WARC/1.0\r\nWARC-Type: conversion\r\n\r\nbody"))
    with pytest.raises(wet.WetError, match="Content-Length"):
        list(wet.pages(path))
