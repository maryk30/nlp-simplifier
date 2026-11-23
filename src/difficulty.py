
import spacy
from textstat import syllable_count
from wordfreq import zipf_frequency


def identify_difficult_words(paragraph):
    difficult_words = []

    for token in paragraph:

        if not token.is_alpha or token.pos_ == "PROPN":
            continue

        word = token.text.lower()
        frequency = zipf_frequency(word, "en")
        length = len(word)
        syllables = syllable_count(word)

        score = 0

        if frequency < 3.5:
            score += 1

        if length >= 9:
            score += 1

        if syllables >= 3:
            score += 1

        if score >= 2:
            difficult_words.append({
                "word": token.text,
                "type": token.pos_,
                "frequency": frequency,
                "length": length,
                "syllables": syllables,
                "score": score
            })

    return difficult_words

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    text = "Photosynthesis is the biochemical process through which green plants convert light energy into chemical energy."

    doc = nlp(text)

    results = identify_difficult_words(doc)

    for word in results:
        print(word)
