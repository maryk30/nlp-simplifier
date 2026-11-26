import difflib
import html
from pathlib import Path

import streamlit as st
import textstat

from bert_replace import replace_with_bert
from difficult_words import identify_difficult_words, sort_words
from main import load_replacements
from preprocess import nlp
from simplify import simplify

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


def stats(text, difficult_count):
    sentences = max(1, textstat.sentence_count(text))
    return {
        "Reading ease": round(textstat.flesch_reading_ease(text), 1),
        "Grade level": round(textstat.flesch_kincaid_grade(text), 1),
        "Words per sentence": round(len(text.split()) / sentences, 1),
        "Difficult words": difficult_count,
    }


def run_pipeline(original):
    simplified = simplify(original, get_replacements())
    replace_words, flag_words = identify_difficult_words(simplified)
    replace_words, flag_words = sort_words(replace_words), sort_words(flag_words)
    final, report, no_match = replace_with_bert(simplified, set(replace_words))

    before_replace, before_flag = identify_difficult_words(original)
    after_replace, after_flag = identify_difficult_words(final)

    return {
        "simplified": simplified,
        "final": final,
        "replace_words": replace_words,
        "flag_words": flag_words,
        "report": report,
        "no_match": no_match,
        "before": stats(original, len(set(before_replace + before_flag))),
        "after": stats(final, len(set(after_replace + after_flag))),
    }


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
                st.session_state["result"] = run_pipeline(text.strip())
                st.session_state["original"] = " ".join(text.split())
            except Exception as exc:
                st.error(f"Something went wrong while simplifying: {exc}")
                st.session_state.pop("result", None)

result = st.session_state.get("result")
if result:
    original = st.session_state["original"]

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.write(original)
    with right:
        st.subheader("Simplified")
        st.markdown(highlight(original, result["final"]), unsafe_allow_html=True)
        st.caption("Highlighted: changed words and new sentence breaks.")

    st.subheader("Difficult words")
    replaced = {}
    for old, new, sentence in result["report"]:
        replaced.setdefault(old.lower(), new)
        replaced.setdefault(nlp(old)[0].lemma_.lower(), new)
    rows = [{"Word": w, "Simpler alternative": replaced.get(w, "—"), "Action": "Replaced" if w in replaced else "Kept (no good alternative)"}
            for w in result["replace_words"]]
    rows += [{"Word": w, "Simpler alternative": "—", "Action": "Kept: key term or name, explain"}
             for w in result["flag_words"]]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.write("No difficult words found.")

    st.subheader("Basic readability checks")
    st.caption("Before vs after. The full six-part evaluation is still to be added.")
    columns = st.columns(4)
    for column, (name, after_value) in zip(columns, result["after"].items()):
        before_value = result["before"][name]
        delta = round(after_value - before_value, 1)
        inverse = name != "Reading ease"
        column.metric(name, after_value, delta, delta_color="inverse" if inverse else "normal")