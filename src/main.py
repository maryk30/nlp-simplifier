
from difficulty import identify_difficult_words
from preprocess import preprocess_text
from sentence import identify_complex_sentences


def main():
    text = """
    Although the monsoon clouds had gathered thickly over the hills since dawn and the wind was already rattling the shutters against the window frames, the team decided to set out along the narrow mountain trail—carrying heavy packs, spare batteries, and waterproof maps—because the delayed weather report warned that the river below would flood before nightfall, leaving no safe way back until morning.
    """

    paragraph, sentences = preprocess_text(text)

    # Identify difficult vocabulary
    difficult_words = identify_difficult_words(paragraph)

    # Identify complex sentences
    complex_sentences = identify_complex_sentences(paragraph)

    print("\nDIFFICULT WORDS")

    for item in difficult_words:
        print(
            f"{item['word']} | "
            f"Frequency: {item['frequency']} | "
            f"Length: {item['length']} | "
            f"Syllables: {item['syllables']} | "
            f"Score: {item['score']}"
        )

    print("\nCOMPLEX SENTENCES")

    for item in complex_sentences:
        print(item["sentence"])
        print(f"Complexity score: {item['score']}")


if __name__ == "__main__":
    main()
