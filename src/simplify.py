import re

from preprocess import nlp

MAX_WORDS = 20
MIN_PART_WORDS = 6

CONJUNCTIONS = {"and": "", "but": "But", "so": "So", "yet": "Yet"}

SENTENCE_START = {"however": "But", "thus": "So"}


def match_case(original, replacement):
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def replace_fixed_words(text, replacements):
    if not replacements:
        return text

    phrases = sorted(replacements, key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:(a|an)\s+)?(" + "|".join(re.escape(p) for p in phrases) + r")\b",
        re.IGNORECASE,
    )

    def substitute(match):
        article, phrase = match.group(1), match.group(2)
        replacement = match_case(phrase, replacements[phrase.lower()])
        if article is None:
            return replacement
        new_article = "an" if replacement[0].lower() in "aeiou" else "a"
        return match_case(article, new_article) + " " + replacement

    return pattern.sub(substitute, text)


def has_subject(text):
    return any(token.dep_ in ("nsubj", "nsubjpass") for token in nlp(text))


def capitalise(text):
    text = text.strip()
    return text[0].upper() + text[1:] if text else text


def split_which(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.lower_ != "which" or token.i == 0 or doc[token.i - 1].text != ",":
            continue
        verb = token.head
        noun = verb.head
        if verb.dep_ != "relcl" or noun.pos_ not in ("NOUN", "PROPN"):
            continue
        left = doc[: token.i - 1].text.strip()
        right = doc[token.i + 1:].text.strip()
        if len(left.split()) < MIN_PART_WORDS or len(right.split()) < MIN_PART_WORDS:
            continue
        determiner = "These" if noun.tag_ in ("NNS", "NNPS") else "This"
        return left.rstrip(".") + ".", f"{determiner} {noun.text.lower()} {right}"
    return None


def split_sentence(sentence):
    if ";" in sentence:
        parts = [p.strip() for p in sentence.split(";") if p.strip()]
        if all(has_subject(p) for p in parts):
            pieces = []
            for part in parts:
                part = capitalise(part).rstrip(".")
                pieces.extend(split_sentence(part + "."))
            return pieces

    if len(sentence.split()) <= MAX_WORDS:
        return [sentence]

    parts = split_which(sentence)
    if parts:
        return [parts[0]] + split_sentence(parts[1])

    pattern = re.compile(r",\s+(and|but|so|yet)\s+", re.IGNORECASE)
    for match in pattern.finditer(sentence):
        left = sentence[: match.start()].strip()
        right = sentence[match.end():].strip()
        if len(left.split()) < MIN_PART_WORDS or len(right.split()) < MIN_PART_WORDS:
            continue
        if not has_subject(right):
            continue
        start = CONJUNCTIONS[match.group(1).lower()]
        right = f"{start} {right}" if start else right
        return [left.rstrip(".") + "."] + split_sentence(capitalise(right))

    return [sentence]


def replace_sentence_start(sentence):
    match = re.match(r"(\w+),\s+", sentence)
    if match and match.group(1).lower() in SENTENCE_START:
        return SENTENCE_START[match.group(1).lower()] + " " + sentence[match.end():]
    return sentence


def simplify_sentences(text):
    doc = nlp(text)
    sentences = []
    for sentence in doc.sents:
        for piece in split_sentence(sentence.text.strip()):
            sentences.append(replace_sentence_start(piece))
    return " ".join(s for s in sentences if s)


def simplify(text, replacements):
    text = re.sub(r"\s+", " ", text).strip()
    text = replace_fixed_words(text, replacements)
    text = simplify_sentences(text)
    return text