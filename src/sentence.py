
def identify_complex_sentences(paragraph):
    complex_sentences = []

    for sentence in paragraph.sents:
        words = [
            token for token in sentence
            if not token.is_punct and not token.is_space
        ]

        word_count = len(words)

        subordinate_clauses = sum(
            1 for token in sentence
            if token.dep_ in ("advcl", "relcl", "ccomp", "acl")
        )

        conjunctions = sum(
            1 for token in sentence
            if token.pos_ == "SCONJ"
        )

        score = 0

        if word_count > 20:
            score += 1

        if subordinate_clauses >= 2:
            score += 1

        if conjunctions >= 2:
            score += 1

        if score >= 2:
            complex_sentences.append({
                "sentence": sentence.text,
                "word_count": word_count,
                "subordinate_clauses": subordinate_clauses,
                "conjunctions": conjunctions,
                "score": score
            })

    return complex_sentences
