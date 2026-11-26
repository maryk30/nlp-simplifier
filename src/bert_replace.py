from transformers import pipeline

from difficult_words import AOA_LIMIT, age_of_acquisition, zipf
from preprocess import nlp

MODEL = "bert-base-uncased"
TOP_K = 30
MIN_ZIPF_GAIN = 0.5

fill_mask = None


def load_model():
    global fill_mask
    if fill_mask is None:
        fill_mask = pipeline("fill-mask", model=MODEL)
    return fill_mask


def predict(sentence_text, masked_text):
    model = load_model()
    return model(f"{sentence_text} {model.tokenizer.sep_token} {masked_text}", top_k=TOP_K)


def masked_sentence(sentence, target, mask):
    return "".join(
        (mask if token.i == target.i else token.text) + token.whitespace_
        for token in sentence
    ).strip()


def substituted_sentence(sentence, target, word):
    return "".join(
        (word if token.i == target.i else token.text) + token.whitespace_
        for token in sentence
    ).strip()


def is_good_candidate(candidate, target, sentence):
    if not candidate.isalpha() or candidate.lower() == target.lower_:
        return False

    position = target.i - sentence.start
    new_token = nlp(substituted_sentence(sentence, target, candidate))[position]
    if new_token.pos_ != target.pos_:
        return False
    if new_token.lemma_.lower() == target.lemma_.lower():
        return False

    if zipf(candidate.lower(), new_token.lemma_.lower()) < zipf(target.lower_, target.lemma_.lower()) + MIN_ZIPF_GAIN:
        return False

    aoa = age_of_acquisition(candidate.lower(), new_token.lemma_.lower())
    return not (aoa is not None and aoa > AOA_LIMIT)


def best_replacement(target, sentence):
    mask = load_model().tokenizer.mask_token
    for prediction in predict(sentence.text, masked_sentence(sentence, target, mask)):
        candidate = prediction["token_str"].strip()
        if is_good_candidate(candidate, target, sentence):
            return candidate
    return None


def match_case(original, replacement):
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def fix_article(doc, replacements):
    for i in list(replacements):
        if i > 0 and doc[i - 1].lower_ in ("a", "an"):
            article = "an" if replacements[i][0].lower() in "aeiou" else "a"
            replacements[i - 1] = match_case(doc[i - 1].text, article)


def replace_with_bert(text, targets):
    doc = nlp(text)
    replacements = {}
    report = []
    no_match = set()

    for sentence in doc.sents:
        for token in sentence:
            if not token.is_alpha or token.lemma_.lower() not in targets:
                continue
            candidate = best_replacement(token, sentence)
            if candidate is None:
                no_match.add(token.lemma_.lower())
                continue
            replacements[token.i] = match_case(token.text, candidate)
            report.append((token.text, candidate, sentence.text.strip()))

    fix_article(doc, replacements)

    output = "".join(
        replacements.get(token.i, token.text) + token.whitespace_ for token in doc
    )
    return output, report, sorted(no_match)