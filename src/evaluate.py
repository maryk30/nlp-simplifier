import re
from functools import lru_cache

import numpy as np
import textstat
from sentence_transformers import SentenceTransformer

import simplify as simplify_module
from bert_replace import (
    fix_article,
    is_good_candidate,
    load_model,
    masked_sentence,
    match_case,
    predict,
    substituted_sentence,
)
from difficult_words import identify_difficult_words, identify_keywords, sort_words
from preprocess import nlp

TARGET_GRADE = 7.0
TARGET_WORDS_PER_SENTENCE = 7.0
WORDS_TOLERANCE = 2.0
MIN_SIMILARITY = 0.80
WORD_SIMILARITY = 0.50
SPLIT_STEPS = [20, 16, 14, 12, 10, 8]
ENCODER_MODEL = "all-MiniLM-L6-v2"

WEIGHTS = {
    "Meaning preservation": 0.25,
    "Information completeness": 0.20,
    "Factual correctness": 0.15,
    "Readability": 0.15,
    "Vocabulary simplification": 0.125,
    "Sentence simplification": 0.125,
}

_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(ENCODER_MODEL)
    return _encoder


def similarity(a, b):
    first, second = encoder().encode([a, b], normalize_embeddings=True)
    return float(np.dot(first, second))


@lru_cache(maxsize=4096)
def word_similarity(a, b):
    return similarity(a.lower(), b.lower())


def defined_terms(doc):
    protected, terms = set(), []
    for i, token in enumerate(doc):
        if token.text != "(" or i + 2 >= len(doc) or doc[i + 2].text != ")":
            continue
        acronym = doc[i + 1].text
        if not (acronym.isupper() and acronym.isalpha() and len(acronym) >= 2):
            continue
        start = i - len(acronym)
        if start < 0:
            continue
        words = doc[start:i]
        initials = "".join(w.text[0] for w in words).upper()
        if initials != acronym:
            continue
        protected.update(range(start, i + 3))
        terms.append(f"{words.text} ({acronym})")
    return protected, terms


def grade_level(text):
    return round(textstat.flesch_kincaid_grade(text), 1)


def words_per_sentence(text):
    sentences = [s for s in nlp(text).sents if s.text.strip()]
    words = [t for t in nlp(text) if t.is_alpha]
    return round(len(words) / max(1, len(sentences)), 1)


def simplify_to_target(original, replacements):
    saved = simplify_module.MAX_WORDS, simplify_module.MIN_PART_WORDS
    simplified, used = original, None
    try:
        for max_words in SPLIT_STEPS:
            simplify_module.MAX_WORDS = max_words
            simplify_module.MIN_PART_WORDS = max(3, min(6, max_words // 3))
            simplified, used = simplify_module.simplify(original, replacements), max_words
            if words_per_sentence(simplified) <= TARGET_WORDS_PER_SENTENCE + WORDS_TOLERANCE:
                break
    finally:
        simplify_module.MAX_WORDS, simplify_module.MIN_PART_WORDS = saved
    return simplified, used


def valid_replacement(target, sentence):
    mask = load_model().tokenizer.mask_token
    rejected = []
    for prediction in predict(sentence.text, masked_sentence(sentence, target, mask)):
        candidate = prediction["token_str"].strip()
        if not is_good_candidate(candidate, target, sentence):
            continue
        word_score = word_similarity(target.text, candidate)
        if word_score < WORD_SIMILARITY:
            rejected.append((candidate, round(word_score, 2)))
            continue
        sentence_score = similarity(sentence.text, substituted_sentence(sentence, target, candidate))
        if sentence_score >= MIN_SIMILARITY:
            return candidate, word_score, sentence_score, rejected
        rejected.append((candidate, round(sentence_score, 2)))
    return None, None, None, rejected


def replace_with_validation(text, targets):
    doc = nlp(text)
    protected, terms = defined_terms(doc)
    replacements, report, kept = {}, [], []

    for sentence in doc.sents:
        for token in sentence:
            if token.i in protected:
                continue
            if not token.is_alpha or token.lemma_.lower() not in targets:
                continue
            candidate, word_score, sentence_score, rejected = valid_replacement(token, sentence)
            if candidate is None:
                kept.append({"word": token.text, "rejected": rejected})
                continue
            replacements[token.i] = match_case(token.text, candidate)
            report.append({"old": token.text, "new": candidate,
                           "word_similarity": round(word_score, 2),
                           "similarity": round(sentence_score, 2),
                           "sentence": sentence.text.strip(), "rejected": rejected})

    fix_article(doc, replacements)
    output = "".join(replacements.get(t.i, t.text) + t.whitespace_ for t in doc)
    return output, report, kept, terms


def sentence_alignment(original, final):
    old = [s.text.strip() for s in nlp(original).sents if s.text.strip()]
    new = [s.text.strip() for s in nlp(final).sents if s.text.strip()]
    if not old or not new:
        return 0.0
    vectors = encoder().encode(old + new, normalize_embeddings=True)
    scores = vectors[: len(old)] @ vectors[len(old):].T
    return float(scores.max(axis=1).mean())


def key_items(text):
    doc = nlp(text)
    keywords = identify_keywords(text)
    items = {t.lemma_.lower() for t in doc if t.lemma_.lower() in keywords}
    items |= {e.text.lower() for e in doc.ents}
    items |= set(re.findall(r"\d+(?:\.\d+)?", text))
    return items


def contains(text, item):
    return re.search(r"\b" + re.escape(item) + r"\b", text, re.IGNORECASE) is not None


def score_readability(before, after):
    if after <= TARGET_GRADE:
        return 100, f"Grade level {before} → {after}, within the target for a 12-year-old (grade {TARGET_GRADE:g} or lower)."
    score = max(0, round(100 - 15 * (after - TARGET_GRADE)))
    return score, f"Grade level {before} → {after}; still above the target of {TARGET_GRADE:g}."


def score_sentences(before, after, max_words):
    gap = abs(after - TARGET_WORDS_PER_SENTENCE)
    score = 100 if gap <= WORDS_TOLERANCE else max(0, round(100 - 10 * (gap - WORDS_TOLERANCE)))
    note = "on target" if gap <= WORDS_TOLERANCE else "the splitting rules could not go further"
    return score, f"Words per sentence {before} → {after} (target about {TARGET_WORDS_PER_SENTENCE:g}; {note}; split limit {max_words})."


def score_vocabulary(before, after, report, kept):
    reduction = 1 - after / before if before else 1
    attempted = len(report) + len(kept)
    valid_share = len(report) / attempted if attempted else 1
    score = round(100 * (0.6 * reduction + 0.4 * valid_share))
    return score, (f"Difficult words {before} → {after}; {len(report)} replacements passed the meaning check, "
                   f"{len(kept)} words kept because no candidate kept the meaning.")


def score_meaning(original, final):
    value = sentence_alignment(original, final)
    score = round(100 * value)
    verdict = "meaning is preserved" if value >= MIN_SIMILARITY else "some meaning may have shifted"
    return score, f"Average sentence similarity {value:.2f}: {verdict}."


def score_completeness(original, final):
    items = key_items(original)
    if not items:
        return 100, "No key terms, names or numbers to check."
    missing = sorted(i for i in items if not contains(final, i))
    score = round(100 * (1 - len(missing) / len(items)))
    detail = f" Missing: {', '.join(missing[:8])}." if missing else " All retained."
    return score, f"{len(items) - len(missing)} of {len(items)} key terms, names and numbers kept.{detail}"


def score_factual(original, final):
    new_numbers = set(re.findall(r"\d+(?:\.\d+)?", final)) - set(re.findall(r"\d+(?:\.\d+)?", original))
    old_ents = {e.text.lower() for e in nlp(original).ents}
    new_ents = {e.text.lower() for e in nlp(final).ents} - old_ents
    introduced = sorted(new_numbers | new_ents)
    score = max(0, 100 - 25 * len(introduced))
    if introduced:
        return score, f"New names or numbers not in the original: {', '.join(introduced)}."
    return score, "No new names or numbers were introduced."


def evaluate(original, final, report, kept, max_words):
    before_r, before_f = identify_difficult_words(original)
    after_r, after_f = identify_difficult_words(final)
    difficult_before = len(set(before_r + before_f))
    difficult_after = len(set(after_r + after_f))

    results = {
        "Meaning preservation": score_meaning(original, final),
        "Information completeness": score_completeness(original, final),
        "Factual correctness": score_factual(original, final),
        "Readability": score_readability(grade_level(original), grade_level(final)),
        "Vocabulary simplification": score_vocabulary(difficult_before, difficult_after, report, kept),
        "Sentence simplification": score_sentences(words_per_sentence(original), words_per_sentence(final), max_words),
    }

    overall = sum(WEIGHTS[name] * score for name, (score, _note) in results.items())
    floor = min(results["Meaning preservation"][0], results["Information completeness"][0])
    overall = round(min(overall, floor + 10))
    return {"scores": results, "overall": overall}


def run(original, replacements):
    original = " ".join(original.split())
    simplified, max_words = simplify_to_target(original, replacements)

    replace_words, flag_words = identify_difficult_words(simplified)
    replace_words, flag_words = sort_words(replace_words), sort_words(flag_words)
    final, report, kept, terms = replace_with_validation(simplified, set(replace_words))
    flag_words = terms + [w for w in flag_words if not any(w.lower() in t.lower().split("(")[-1].lower() or w.lower() in t.lower().split() for t in terms)]

    evaluation = evaluate(original, final, report, kept, max_words)
    return {
        "original": original,
        "simplified": simplified,
        "final": final,
        "replace_words": replace_words,
        "flag_words": flag_words,
        "report": report,
        "kept": kept,
        "evaluation": evaluation,
    }