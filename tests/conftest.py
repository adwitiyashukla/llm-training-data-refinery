import gzip
import json
import random
from pathlib import Path

import pytest

from refinery import wet
from refinery.config import parse
from refinery.text import hash64

SUBJECTS = [
    "The city council", "A local farmer", "Our science teacher", "The research team", "My neighbor",
    "The new library", "A small bakery", "The football club", "The history museum", "A group of students",
    "The county nurse", "An old fisherman", "The music school", "A young engineer", "The hiking club",
    "The weather office", "A retired pilot", "The garden society", "The town mayor", "A careful editor",
    "The night shift", "A visiting author", "The fire chief", "Two park rangers", "The chess team",
    "A quiet monk", "The bus driver", "The school board", "A street vendor", "The radio host",
]
VERBS = [
    "described", "reviewed", "announced", "repaired", "studied", "visited", "measured", "painted",
    "explained", "planned", "photographed", "cleaned", "mapped", "tested", "opened", "watched",
    "admired", "rebuilt", "inspected", "sketched", "praised", "questioned", "funded", "restored",
    "moved", "guarded", "noticed", "shared", "covered", "toured",
]
ADJECTIVES = [
    "quiet", "narrow", "busy", "ancient", "bright", "muddy", "wooden", "crowded", "distant", "famous",
    "hidden", "modern", "rusty", "sunny", "tiny", "wide", "broken", "golden", "humble", "steep",
    "frozen", "noisy", "elegant", "empty", "fragile", "gentle", "heavy", "lonely", "massive", "patient",
    "plain", "proud", "rapid", "rough", "silent", "sturdy", "tall", "warm", "wild", "young",
]
NOUNS = [
    "bridge", "garden", "harbor", "market", "station", "orchard", "tunnel", "chapel", "factory", "school",
    "meadow", "village", "canal", "lighthouse", "theater", "warehouse", "hospital", "stadium", "forest", "valley",
    "barn", "castle", "clinic", "dock", "fountain", "gallery", "highway", "island", "kitchen", "ladder",
    "mill", "office", "pier", "quarry", "ranch", "shelter", "tower", "vineyard", "workshop", "yard",
    "archive", "bakery", "cabin", "dam", "estate", "ferry", "glacier", "hotel", "inn", "jetty",
    "kiosk", "lodge", "mansion", "nursery", "observatory", "palace", "reservoir", "studio", "temple", "zoo",
]
PREPS = ["near", "beside", "behind", "across from", "under", "along", "inside", "past", "around", "above",
         "below", "next to"]
TIMES = ["yesterday", "last week", "on Monday", "in March", "at dawn", "this spring", "after lunch",
         "before sunset", "in the evening", "last year", "on Friday", "early today", "in July", "this morning",
         "last month", "on Sunday", "in October", "at noon"]


def sentence(rng):
    a, b, c = rng.sample(ADJECTIVES, 3)
    n1, n2, n3 = rng.sample(NOUNS, 3)
    form = rng.randrange(4)
    if form == 0:
        return (f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} the {a} {n1} {rng.choice(PREPS)} the {b} {n2} "
                f"{rng.choice(TIMES)}.")
    if form == 1:
        when = rng.choice(TIMES)
        return (f"{when[0].upper()}{when[1:]}, {rng.choice(SUBJECTS).lower()} {rng.choice(VERBS)} a {a} {n1} "
                f"and the {n2} {rng.choice(PREPS)} the {n3}.")
    if form == 2:
        return f"The {n1} {rng.choice(PREPS)} the {a} {n2} was {b} and {c} {rng.choice(TIMES)}."
    return (f"{rng.choice(SUBJECTS)} said that the {a} {n1} of the {n2} {rng.choice(VERBS)} the {n3} "
            f"{rng.choice(TIMES)}.")


FOREIGN = {
    "fr": (["Le conseil municipal", "Une famille du quartier", "Notre equipe de recherche", "Le professeur"],
           ["a presente", "a decouvert", "a explique", "a construit"],
           ["un nouveau projet pour la ville", "une petite maison en bois", "le jardin public", "la vieille ecole"],
           ["pendant les vacances.", "avant la fin de la semaine.", "avec les enfants du village."]),
    "de": (["Der Stadtrat", "Eine Familie aus der Stadt", "Unser Forschungsteam", "Der Lehrer"],
           ["hat vorgestellt", "hat entdeckt", "hat erklaert", "hat gebaut"],
           ["ein neues Projekt fuer die Stadt", "ein kleines Haus aus Holz", "den oeffentlichen Garten",
            "die alte Schule"],
           ["in den Ferien.", "vor dem Ende der Woche.", "mit den Kindern aus dem Dorf."]),
    "es": (["El consejo de la ciudad", "Una familia del barrio", "Nuestro equipo de investigacion", "El profesor"],
           ["presento", "descubrio", "explico", "construyo"],
           ["un nuevo proyecto para la ciudad", "una casa pequena de madera", "el parque publico",
            "la escuela antigua"],
           ["durante las vacaciones.", "antes del fin de semana.", "para los vecinos del pueblo."]),
}
CC_TAG = {"en": "eng", "fr": "fra", "de": "deu", "es": "spa"}
BLOCKED = ("blockedtoken", "blockedsite")


def english(rng, lines=8):
    return "\n".join(" ".join(sentence(rng) for _ in range(3)) for _ in range(lines))


def foreign(rng, lang, lines=8):
    subjects, verbs, objects, tails = FOREIGN[lang]
    return "\n".join(
        " ".join(f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(tails)}"
                 for _ in range(3))
        for _ in range(lines)
    )


def edit(rng, text, rate):
    lines = []
    for line in text.split("\n"):
        words = line.split()
        for i in range(len(words)):
            if rng.random() < rate:
                words[i] = rng.choice(NOUNS)
        lines.append(" ".join(words))
    return "\n".join(lines)


def make_crawl(base, crawl_id, shards=3, per_shard=40, seed=7, earlier=None):
    rng = random.Random(seed)
    base = Path(base)
    rels, truth, made = [], {}, []
    for shard in range(shards):
        rel = f"crawl-data/{crawl_id}/segments/{shard:04d}/wet/part-{shard:05d}.warc.wet.gz"
        items = []
        for k in range(per_shard):
            host = f"site{rng.randrange(12)}.example"
            url = f"https://{host}/{crawl_id.lower()}/{shard}/{k}"
            roll = rng.random()
            kind = "english"
            if earlier and roll < 0.15:
                kind, text = "copy_prior", rng.choice(earlier)
            elif earlier and roll < 0.25:
                kind, text = "near_prior", edit(rng, rng.choice(earlier), 0.02)
            elif roll < 0.33:
                lang = rng.choice(sorted(FOREIGN))
                kind, text = f"foreign_{lang}", foreign(rng, lang)
            elif roll < 0.37:
                kind, text = "spam", "\n".join(["Buy cheap watches online today best price free shipping"] * 30)
            elif roll < 0.40:
                kind, text = "short", "Click here to continue reading this page."
            elif roll < 0.43:
                kind, text = "code", english(rng, 4) + "\nfunction load() { return window.data; }\n" + english(rng, 2)
            elif roll < 0.45:
                kind, text = "lorem", english(rng, 3) + "\nLorem ipsum dolor sit amet, consectetur adipiscing elit.\n"
            elif roll < 0.47:
                kind, text = "blocked", english(rng)
                url = f"https://blockedtoken-deals.example/{shard}/{k}"
            elif roll < 0.52 and made:
                kind, text = "exact", rng.choice(made)
            elif roll < 0.57 and made:
                kind, text = "near", edit(rng, rng.choice(made), 0.02)
            elif roll < 0.60:
                kind = "pii"
                a, b = rng.sample(ADJECTIVES, 2)
                n1, n2 = rng.sample(NOUNS, 2)
                text = english(rng, 6) + (f"\nWrite to {a}.{n1}{k}@example.org about the {b} {n2} or call "
                                          f"(513) 555-{1000 + k} from the office at 8.8.{shard}.{k} before noon.")
            else:
                text = english(rng)
                if rng.random() < 0.5:
                    text = f"Welcome to the {host} community news archive and events board\n" + text
                made.append(text)
            lang = kind.removeprefix("foreign_") if kind.startswith("foreign_") else "en"
            items.append((url, text, CC_TAG[lang]))
            truth[url] = kind
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        wet.write(path, items)
        rels.append(rel)
    listing = base / "crawl-data" / crawl_id / "wet.paths.gz"
    with gzip.open(listing, "wt", encoding="utf-8") as f:
        f.write("\n".join(rels) + "\n")
    return truth, made


def raw_settings(tmp_path, base, crawls, workers=1, **overrides):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    blocklist = Path(tmp_path) / "blocklist.txt"
    blocklist.write_text("".join(f"{hash64(t):016x}\n" for t in BLOCKED), encoding="utf-8")
    raw = {
        "run": {"data_dir": (Path(tmp_path) / "data").as_posix(), "workers": workers, "batch_docs": 100},
        "crawls": [{"id": c, "shards": n} for c, n in crawls],
        "fetch": {"base_url": Path(base).resolve().as_uri() + "/", "retries": 0, "timeout": 10.0},
        "url": {"blocklist": blocklist.resolve().as_posix()},
        "dedup": {"bloom_capacity": 100000, "bloom_fpr": 0.001},
        "eval": {"shards": 2},
    }
    for key, value in overrides.items():
        section, name = key.split("__")
        raw.setdefault(section, {})[name] = value
    return raw


def settings(tmp_path, base, crawls, workers=1, **overrides):
    return parse(raw_settings(tmp_path, base, crawls, workers, **overrides))


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(value)


def write_config(path, raw):
    lines = []
    for section, values in raw.items():
        if section == "crawls":
            continue
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in values.items())
        lines.append("")
    for crawl in raw["crawls"]:
        lines.append("[[crawls]]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in crawl.items())
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def web(tmp_path):
    base = tmp_path / "web"
    truth, made = make_crawl(base, "CC-MAIN-2099-01", shards=3, per_shard=40, seed=7)
    return base, truth, made
