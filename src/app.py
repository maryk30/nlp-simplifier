import difflib
import html
import re
from pathlib import Path

import streamlit as st

from evaluate import TARGET_GRADE, TARGET_WORDS_PER_SENTENCE, run
from main import load_replacements

REPLACEMENTS = Path("data/replacement_words.csv")
MIN_WORDS = 15
MAX_CHARS = 5000


@st.cache_data
def get_replacements():
    return load_replacements(REPLACEMENTS)


def validate(text):
    text = text.strip()
    if not text:
        return "Please enter some text."
    if len(text) > MAX_CHARS:
        return f"Please keep the text under {MAX_CHARS} characters (yours has {len(text)})."
    if len(text.split()) < MIN_WORDS:
        return f"Please enter at least {MIN_WORDS} words, so the readability scores are meaningful."
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return "This tool works on English text only."
    return None


def short_note(note):
    change = re.search(r"(-?\d+(?:\.\d+)?) → (-?\d+(?:\.\d+)?)", note)
    if change:
        return f"{change.group(1)} → {change.group(2)}"
    kept = re.search(r"(\d+) of (\d+)", note)
    if kept:
        return f"{kept.group(1)} of {kept.group(2)} kept"
    value = re.search(r"\d\.\d+", note)
    if value:
        return value.group(0)
    if note.startswith("No new"):
        return "0 new"
    return f"{len(note.split(':', 1)[-1].split(','))} new"


def highlight(before, after):
    old, new = before.split(), after.split()
    parts = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        chunk = html.escape(" ".join(new[j1:j2]))
        if not chunk:
            continue
        if tag == "equal":
            parts.append(chunk)
        else:
            parts.append(f'<mark style="background:#EBD3CB;padding:0 2px">{chunk}</mark>')
    return " ".join(parts)


st.set_page_config(page_title="Educational Text Simplifier", layout="wide")
st.title("Educational Text Simplifier")
st.caption("Simplifies an educational paragraph while keeping its key terms.")

text = st.text_area("Educational text", height=220, placeholder="Paste a paragraph here…")

if st.button("Simplify", type="primary"):
    error = validate(text)
    if error:
        st.error(error)
    else:
        with st.spinner("Simplifying… the first run downloads the BERT model, which takes a minute."):
            try:
                st.session_state["result"] = run(text.strip(), get_replacements())
            except Exception as exc:
                st.error(f"Something went wrong while simplifying: {exc}")
                st.session_state.pop("result", None)

result = st.session_state.get("result")
if result:
    original = result["original"]

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.write(original)
    with right:
        st.subheader("Simplified")
        st.markdown(highlight(original, result["final"]), unsafe_allow_html=True)
        st.caption("Highlighted: changed words and new sentence breaks.")

    st.subheader("Difficult words")
    rows = [{"Word": r["old"], "Simpler alternative": r["new"], "Word similarity": r["word_similarity"],
             "Sentence similarity": r["similarity"], "Action": "Replaced"} for r in result["report"]]
    rows += [{"Word": k["word"], "Simpler alternative": "—", "Word similarity": None, "Sentence similarity": None,
              "Action": "Kept: no candidate kept the meaning"} for k in result["kept"]]
    rows += [{"Word": w, "Simpler alternative": "—", "Word similarity": None, "Sentence similarity": None,
              "Action": "Kept: key term or name, explain"} for w in result["flag_words"]]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.write("No difficult words found.")

    evaluation = result["evaluation"]
    st.subheader(f"Evaluation · overall {evaluation['overall']}/100")
    st.caption(f"Targets: grade level {TARGET_GRADE:g} or lower (about age 12), "
               f"about {TARGET_WORDS_PER_SENTENCE:g} words per sentence.")
    names = list(evaluation["scores"])
    for row_names in (names[:3], names[3:]):
        for column, name in zip(st.columns(3), row_names):
            score, note = evaluation["scores"][name]
            column.metric(name, f"{score}/100")
            column.caption(short_note(note))