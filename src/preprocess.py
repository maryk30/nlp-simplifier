import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    paragraph = nlp(text)
    sentences = []

    for word in paragraph:
        sentences.append(word.text)

    return paragraph, sentences
