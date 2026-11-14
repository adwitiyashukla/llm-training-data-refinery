# llm-training-data-refinery

[![ci](https://github.com/adwitiyashukla/llm-training-data-refinery/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/llm-training-data-refinery/actions/workflows/ci.yml)

A pipeline that turns raw Common Crawl web text into a clean, deduplicated pretraining corpus for
language models, the same kind of work that goes into datasets like FineWeb. I built it to see what
actually happens between "download the web" and "train on it", and how far one laptop can go.

It reads the WET files Common Crawl publishes, runs the published FineWeb filtering rules, removes
duplicates at four levels, masks personal details, and writes Parquet shards with a manifest. Every
dropped page is logged with the rule that dropped it and the value that failed. A run can be killed
at any point and started again, and it continues where it stopped.

Live demo: https://huggingface.co/spaces/adwitiyashukla/llm-training-data-refinery

## One real run

Two Common Crawl snapshots on my laptop, an HP OmniBook with an Intel Core 5 120U and 8 GB of RAM,
Windows 11, Python 3.13.7, 6 workers. I ran it on 50 WET files spread evenly across the June 2026
crawl and 25 across the August 2026 crawl, about 1.5 million pages.

| | June 2026 | August 2026 |
| --- | --- | --- |
| Crawl | CC-MAIN-2026-25 | CC-MAIN-2026-34 |
| WET files | 50 | 25 |
| Compressed input | 2.93 GiB | 1.49 GiB |
| Pages read | 1,024,667 | 520,862 |
| Pages kept | 93,447 | 47,179 |
| Share kept | 9.12% | 9.06% |
| Words kept | 86,020,254 | 44,612,345 |
| Corpus on disk | 232.8 MB | 120.3 MB |
| Wall time | 42.1 min | 24.9 min |
| Pages per second | 405 | 349 |

Peak memory was 253 MB per worker and 1,954 MB for the main process, so the whole run fits in 8 GB.
Each raw file is deleted as soon as it is processed, so there were never more than six on disk at
once, and the data folder ended at about 560 MB.

### Where the pages go

June, step by step:

| Step | Pages dropped | Pages left | Share left |
| --- | --- | --- | --- |
| Pages read | | 1,024,667 | 100.00% |
| URL blocklist | 6,159 | 1,018,508 | 99.40% |
| Language ID, English at score 0.65 | 659,724 | 358,784 | 35.01% |
| Gopher repetition | 146,966 | 211,818 | 20.67% |
| Gopher quality | 62,150 | 149,668 | 14.61% |
| C4 | 3,990 | 145,678 | 14.22% |
| FineWeb | 44,735 | 100,943 | 9.85% |
| Same URL | 527 | 100,416 | 9.80% |
| Same text | 1,180 | 99,236 | 9.68% |
| Near duplicate | 2,640 | 96,596 | 9.43% |
| Under 50 words after line dedup | 3,149 | 93,447 | 9.12% |

Language ID removes almost two thirds of the pages. After English, the biggest groups are Russian,
German, Japanese, French, Spanish and Chinese.

Most of the rest goes at the repetition and quality rules. The top reasons are more than 30 percent
of a page's lines being repeats (80,540 pages), fewer than 80 percent of words containing a letter
(54,885) and fewer than 12 percent of lines ending in punctuation (34,012). FineWeb tuned these
rules on text pulled out of the HTML by trafilatura, which strips most menus and footers first. WET
text keeps them, so a page with a real article and a long menu still repeats lines and fails more
often. The C4 step also removed 4,823,654 lines of under three words, like menu items and buttons,
from the pages that passed.

## Deduplication

Four levels, cheapest first:

1. Same URL after normalizing the host, the trailing slash and the fragment.
2. Same text after lowercasing, mapping digits to 0 and dropping punctuation.
3. Near duplicates with MinHash. Pages are cut into 5 word shingles, hashed 112 times and split into
   14 bands of 8 rows. Any shared band makes a candidate pair, the pair is kept only if the two
   signatures agree on at least 70 percent of hashes, and union-find groups the pairs. One page per
   group survives.
4. Repeated lines. Every line is hashed and checked against a Bloom filter of all lines seen before.
   Lines already seen are removed, and a page that drops under 50 words afterwards is removed too.

In June that removed 527 repeated URLs, 1,180 exact copies and 2,640 near duplicates in 1,122
groups. The biggest group had 138 pages. Line dedup took out 1,413,416 of 5,574,690 lines (25.35
percent) from the pages that stayed.

August ran after June and was checked against June's saved state. 1,171 URLs, 495 exact texts and
1,165 near duplicates were already in the June output, 2,831 pages or 5.4 percent of what got past
the August filters. Both crawls here are small slices, 75 of 200,000 files, so this mostly shows the
cross-crawl check working rather than how much two full crawls overlap.

### Is the near-duplicate step right

To check it I ran an exact similarity join on the first 20 June shards, 40,000 documents, and took
every pair with Jaccard similarity of at least 0.75 on the real shingle sets as a true near
duplicate. Then I compared that with what the pipeline removed, for a few values of the signature check.

| Signature check | Pages removed | Kept page within Jaccard 0.75 | Kept page within 0.5 | Duplicates caught |
| --- | --- | --- | --- | --- |
| none | 1,068 | 51.9% | 90.4% | 98.5% |
| 0.60 | 958 | 58.3% | 97.5% | 98.5% |
| 0.70, the default | 818 | 69.8% | 99.6% | 97.6% |
| 0.80 | 593 | 92.8% | 100.0% | 85.2% |

Duplicates caught is the share of the 675 pages with an earlier near duplicate that got removed.
Without the signature check, LSH candidates chain together through union-find and pull in pages
that are only loosely related. At 0.70 the pipeline keeps almost all of the recall, and 99.6 percent of
removed pages share at least half their shingles with the page kept in their place.

The chance that two pages become candidates follows the theory for 14 bands of 8 rows. In the same
sample, pairs with Jaccard between 0.80 and 0.85 became candidates 98.0 percent of the time against
97.0 predicted, and pairs between 0.50 and 0.55 only 6.4 percent of the time against 7.9 predicted.

### Bloom filter

The line filter is sized for 20 million lines at a 0.1 percent false positive rate, which gives
287,551,808 bits and 10 hashes. During June every lookup was also checked against exact hashing: 0
false positives in 4,484,052 new lines and 0 false negatives. After both crawls the filter was 19.5
percent full, which puts the expected false positive rate at 7.8e-08.

## Checking it against other tools

The first three checks ran once on a separate sample of 12 real WET files, 247,072 pages. The last
one comes from the June run.

| Part | Compared with | Result |
| --- | --- | --- |
| WET parser | warcio | 0 mismatches in 247,072 records |
| Filters | datatrove, the library behind FineWeb | same decision on 97.3% of 15,000 English pages |
| MinHash | datasketch with 112 permutations | mean absolute error 0.0275 against exact Jaccard, datasketch 0.0302, on 10,863 page pairs |
| Language ID | Common Crawl's own language tag | 96.7% agreement on 1,017,908 pages |

Per rule, the filters agree 99.99 percent of the time for FineWeb, 99.95 for C4, 96.65 for Gopher
repetition and 94.67 for Gopher quality. The gaps come from two places. datatrove splits words with
spaCy and I use a regular expression, which moves the share of words with a letter on pages full of
URLs or numbers. And for the Gopher duplicate n-gram rule, I count every repeated n-gram the way the
Gopher paper defines it, while datatrove's loop skips some, so my version drops a few more pages
there.

PII masking replaced 17,410 emails, 665 public IP addresses and 18,472 phone numbers across both
crawls.

## What a run does

1. Download. Workers stream WET files from data.commoncrawl.org with retries and resume, one file
   each, and delete every raw file once it is processed.
2. Filter. Each page goes through the URL blocklist, fastText language ID, the Gopher repetition and
   quality rules, the C4 rules and the FineWeb rules, in that order. The first failing rule is the
   drop reason, and it lands in a Parquet ledger together with the value that failed.
3. Deduplicate. The four levels above.
4. Mask. Emails, public IP addresses and phone numbers are replaced with placeholders.
5. Publish. Parquet shards, a manifest with the SHA-256 of every input and output file, and a run
   card with the funnel and all the checks above.

## Crash safety and adding a crawl

Every file the pipeline writes goes to a temporary name first and is renamed into place after an fsync,
and every stage leaves a small done marker per shard. Killing a run at any point and starting it
again picks up from the last finished shard. The tests kill a run on purpose at four different
stages, with one worker and with two, restart it, and check that the output files are byte for byte
the same as a run that was never interrupted. They also check that 1 worker and 3 workers give the
same output.

The dedup state (URL hashes, text hashes, MinHash band keys of kept pages and the Bloom filter) is
saved after every crawl. Adding a crawl to `config.toml` and running again processes only the new
crawl and deduplicates it against everything before it, without touching the earlier output. If a
filter setting changes between crawls, the pipeline refuses to continue rather than mixing two rule sets
in one corpus.

## Running it

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
refinery run
```

Other commands:

```
refinery status                                      how far each crawl got
refinery explain page.txt --url https://example.com  every rule on one page, with its value and limit
refinery export --out space                          copy the run numbers into the demo folder
```

Output lands in `data/`: the corpus in `data/corpus`, the drop ledger in `data/ledger`, and the run
card, metrics and manifest in `data/reports`.

## Configuration

`config.toml` holds every setting and is validated on load, so a typo or a value out of range fails
at startup with the key named. The crawls to read and how many WET files to take from each, the
worker count, every filter threshold, the MinHash shingle size, bands and rows, the Bloom filter
capacity, the PII switches and the evaluation sample all live there. The filter thresholds default
to the published Gopher, C4 and FineWeb values.

## Tests

54 tests with pytest. The rules are tested one by one, the n-gram statistics against a brute force
version, and the similarity join, union-find and the LSH candidate rate against brute force or
theory. The whole pipeline runs against generated WET files with planted duplicates, near
duplicates, foreign pages, spam and personal details. CI runs the suite on Ubuntu and Windows with
Python 3.11 and 3.13.

## Credits

The URL blocklist is built from the English word list of the LDNOOBW project
(https://github.com/LDNOOBW), licensed CC BY 4.0. This repo stores only xxHash values of the entries.
Language ID uses the fastText lid.176 model that ships with fast-langdetect.
