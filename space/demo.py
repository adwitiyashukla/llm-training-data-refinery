import json

from refinery.config import blocklist_bytes, parse
from refinery.langid import LangId
from refinery.rules import Rules, parse_blocklist

GROUPS = (("url", "URL"), ("lang", "language"), ("rep", "Gopher repetition"), ("gopher", "Gopher quality"),
          ("c4", "C4"), ("fineweb", "FineWeb"))


class BrowserModel:
    def __init__(self, predict):
        self.js_predict = predict

    def predict(self, line, k=1):
        label, prob = self.js_predict(line, k)
        return [label], [prob]


def group_of(name):
    return next((label for prefix, label in GROUPS if name.startswith(prefix)), "other")


def load_rules(predict):
    cfg = parse({"crawls": [{"id": "CC-MAIN-2026-25", "shards": 1}]})
    return Rules(cfg, parse_blocklist(blocklist_bytes(cfg)), LangId(BrowserModel(predict)))


def judge(rules, url, text):
    verdict = rules.judge(url, text, explain=True)
    gone = {index for index, _ in verdict.removed}
    checks = [{"stage": group_of(c.name), "rule": c.name, "value": c.value, "limit": c.limit, "ok": c.ok}
              for c in verdict.checks]
    lines = [[line, i in gone] for i, line in enumerate(text.split("\n"))]
    return json.dumps({"reason": verdict.reason, "lang": verdict.lang, "score": verdict.score,
                       "checks": checks, "lines": lines})
