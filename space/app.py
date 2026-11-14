import json
from pathlib import Path

import pandas as pd
import streamlit as st

from refinery.config import parse
from refinery.langid import LangId
from refinery.pipeline import banned_words
from refinery.rules import Rules

HERE = Path(__file__).parent
GROUPS = (("url", "URL"), ("lang", "language"), ("rep", "Gopher repetition"), ("gopher", "Gopher quality"),
          ("c4", "C4"), ("fineweb", "FineWeb"))
EXAMPLES = {
    "News article": (
        "https://example.com/news/library-reading-room",
        "City Library Opens a New Reading Room\n"
        "The city library opened a new reading room on Monday after almost a year of work on the old north wing.\n"
        "The room has forty desks, large windows that face the park, and a quiet corner with books for young "
        "children and their parents.\n"
        "Library staff said the project was paid for by a state grant and by donations from more than two hundred "
        "local families.\n"
        "The head librarian thanked the volunteers who helped move about six thousand books into the new shelves "
        "during the summer.\n"
        "Visitors can use the room every day from nine in the morning until eight at night, and the first hour of "
        "printing is free for students.\n"
        "A small opening event with music and free coffee will be held in the main hall on Saturday afternoon.",
    ),
    "Menu page": (
        "https://example.com/home",
        "Home\nAbout us\nServices\nPricing\nBlog\nCareers\nContact\nLog in\nSign up\nFollow us\n"
        "Privacy\nTerms\nHelp center\nSitemap\nWelcome to our website.",
    ),
    "Repeated spam": (
        "https://example.com/deals",
        "\n".join(["Best deals on running shoes today, free shipping on every single order."] * 12),
    ),
    "Not English": (
        "https://example.com/es/biblioteca",
        "La biblioteca de la ciudad tiene una nueva sala de lectura. La sala tiene cuarenta mesas y ventanas "
        "grandes que dan al parque.\n"
        "Hay una zona tranquila llena de libros para las familias y un espacio para estudiar en grupo.\n"
        "La sala abre de lunes a viernes, desde las nueve hasta las ocho de la noche, y la entrada es gratis para "
        "todas las personas.",
    ),
    "Code snippet": (
        "https://example.com/blog/python-dicts",
        "In this short post I want to show how a dictionary works in Python and why it is so useful for counting "
        "things.\n"
        "A dictionary maps each key to a value, and looking up a key is fast even when the dictionary is very large.\n"
        "Here is a tiny example that counts colors in a list: counts = {'red': 2, 'blue': 1}.\n"
        "You can add a new key at any time, and you can loop over the keys to print every count one by one.\n"
        "Most people learn this in their first week of Python, and they keep using it for the rest of their career.",
    ),
    "Short page": (
        "https://example.com/missing",
        "Page not found.\nThe page you are looking for does not exist.",
    ),
}

st.set_page_config(page_title="llm-training-data-refinery", layout="wide")


@st.cache_resource
def load_rules():
    cfg = parse({"crawls": [{"id": "CC-MAIN-2026-25", "shards": 1}]})
    return Rules(cfg, banned_words(cfg), LangId())


@st.cache_data
def load_runs():
    return {p.stem.removeprefix("metrics-"): json.loads(p.read_text()) for p in sorted(HERE.glob("metrics-*.json"))}


def group_of(name):
    return next((label for prefix, label in GROUPS if name.startswith(prefix)), "other")


def run_tab(runs):
    if not runs:
        st.info("No run numbers bundled with this Space yet.")
        return
    crawl = st.selectbox("Crawl", list(runs))
    m = runs[crawl]
    pages = m["input"]["pages"]
    cols = st.columns(4)
    cols[0].metric("Pages read", f"{pages:,}")
    cols[1].metric("Pages kept", f"{m['published']['docs']:,}", f"{100 * m['kept_share']:.1f}% of pages")
    cols[2].metric("Words kept", f"{m['published']['words']:,}")
    cols[3].metric("Pages per second", f"{m['pages_per_second']:.0f}")
    funnel = pd.DataFrame(m["funnel"])
    funnel["remaining share"] = (100 * funnel["remaining"] / pages).round(2)
    st.subheader("Funnel")
    st.bar_chart(funnel.set_index("step")["remaining"], horizontal=True)
    st.dataframe(funnel, hide_index=True, width="stretch")
    reasons = pd.DataFrame(list(m["reasons"].items()), columns=["reason", "pages"])
    st.subheader("Why pages were dropped")
    st.dataframe(reasons, hide_index=True, width="stretch")
    lsh = (m.get("eval") or {}).get("lsh")
    if lsh:
        st.subheader("Near-duplicate check against an exact similarity join")
        st.write(f"{lsh['sample_docs']:,} documents checked. {100 * lsh['removed_close']:.1f}% of removed pages are "
                 f"within Jaccard {lsh['jaccard']} of the page kept in their place, and "
                 f"{100 * lsh['doc_recall']:.1f}% of pages with an earlier near duplicate were removed.")
        curve = pd.DataFrame(lsh["curve"])
        if not curve.empty:
            curve["bin"] = curve["low"].map(lambda v: f"{v:.2f}")
            st.line_chart(curve.set_index("bin")[["measured", "theory"]])


def try_tab():
    if "text" not in st.session_state:
        st.session_state["url"], st.session_state["text"] = EXAMPLES["News article"]
    st.write("Pick an example or paste your own text. Every rule runs on it and you can see which one fails first.")
    cols = st.columns(len(EXAMPLES))
    for col, (name, (url, text)) in zip(cols, EXAMPLES.items(), strict=True):
        if col.button(name, width="stretch"):
            st.session_state["url"] = url
            st.session_state["text"] = text
    url = st.text_input("URL", key="url")
    text = st.text_area("Page text", key="text", height=260)
    if not text.strip():
        return
    verdict = load_rules().judge(url, text, explain=True)
    result = "kept" if not verdict.reason else f"dropped by {verdict.reason}"
    st.markdown(f"**{result}**, language `{verdict.lang}` ({verdict.score:.3f})")
    checks = verdict.checks
    st.dataframe(pd.DataFrame({
        "stage": [group_of(c.name) for c in checks],
        "rule": [c.name for c in checks],
        "value": [round(c.value, 4) for c in checks],
        "limit": [round(c.limit, 4) for c in checks],
        "result": ["pass" if c.ok else "fail" for c in checks],
    }), hide_index=True, width="stretch")
    gone = {index for index, _ in verdict.removed}
    lines = [f"[removed] {line}" if i in gone else line for i, line in enumerate(text.split("\n"))]
    st.code("\n".join(lines), language=None, wrap_lines=True)


st.title("llm-training-data-refinery")
st.write("This project turns raw Common Crawl web text into clean training text for language models. "
         "Here you can see the numbers from a real run and try the same quality rules on any text.")
summary, rules = st.tabs(["Run summary", "Try the rules"])
with summary:
    run_tab(load_runs())
with rules:
    try_tab()
