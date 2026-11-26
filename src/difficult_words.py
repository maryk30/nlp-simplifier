from pathlib import Path

import numpy as np
import pandas as pd
import yake
from wordfreq import zipf_frequency

from preprocess import preprocess_text

RARE_ZIPF = 3.6
AOA_LIMIT = 10.0
AOA_PATH = Path("data/AoA_ratings_Kuperman_et_al_BRM.xlsx")


def load_aoa(path):
    if not path.exists():
        return {}
    table = pd.read_excel(path)[["Word", "Rating.Mean"]].dropna()
    table["Rating.Mean"] = pd.to_numeric(table["Rating.Mean"], errors="coerce")
    table = table.dropna()
    return dict(zip(table["Word"].astype(str).str.lower(), table["Rating.Mean"]))


AOA = load_aoa(AOA_PATH)


def zipf(word, word_lemma):
    if word != word_lemma:
        zipf_word = zipf_frequency(word, "en")
        zipf_lemma = zipf_frequency(word_lemma, "en")
        z_word = 10 ** zipf_word if zipf_word > 0 else 0
        z_lemma = 10 ** zipf_lemma if zipf_lemma > 0 else 0
        raw = z_word + z_lemma
        if raw == 0:
            return 0.0
        freq = np.log10(raw)
        freq = np.round(freq, decimals=2)
    else:
        freq = zipf_frequency(word, "en")

    return freq


def age_of_acquisition(word, word_lemma):
    if word in AOA:
        return AOA[word]
    if word_lemma in AOA:
        return AOA[word_lemma]
    return None


def identify_keywords(text):
    n_words = len(text.split())
    top_k = max(3, min(20, n_words // 15))

    keyword_extractor = yake.KeywordExtractor(lan="en", n=1, window_size=6, top=top_k)
    keyphrases = keyword_extractor.extract_keywords(text)

    keyword_words = set()
    for phrase, _score in keyphrases:
        for word in phrase.lower().split():
            keyword_words.add(word)

    return keyword_words


def is_difficult(freq, aoa):
    if aoa is not None:
        return aoa > AOA_LIMIT
    return freq < RARE_ZIPF


def identify_difficult_words(text):

    replace_words = []
    flag_words = []

    paragraph, _sentences = preprocess_text(text)
    keyword_words = identify_keywords(text)

    for token in paragraph:

        if not token.is_alpha:
            continue

        word_lemma = token.lemma_.lower()
        word = token.text.lower()

        freq = zipf(word, word_lemma)
        aoa = age_of_acquisition(word, word_lemma)
        if not is_difficult(freq, aoa):
            continue

        if token.pos_ == "PROPN":
            flag_words.append(token.text)
        elif word in keyword_words or word_lemma in keyword_words:
            flag_words.append(word_lemma)
        else:
            replace_words.append(word_lemma)

    return replace_words, flag_words

def sort_words(word_list: list):
    word_list.sort()

    sorted_words = []

    for word in word_list:
        if word not in sorted_words:
            sorted_words.append(word)

    return sorted_words

def main():
    text = Path("input/test_data.txt").read_text(encoding="utf-8")
    replaceable_words, non_replaceable_words = identify_difficult_words(text)
    non_replaceable_words = sort_words(non_replaceable_words)
    replaceable_words = sort_words(replaceable_words)
    with open("input/replace.txt", "w", encoding="utf-8") as file:
        file.writelines(f"{word}\n" for word in replaceable_words)
    with open("input/keywords.txt", "w", encoding="utf-8") as file:
        file.writelines(f"{word}\n" for word in non_replaceable_words)
