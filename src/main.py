import csv
import shutil
from pathlib import Path

from bert_replace import replace_with_bert
from difficult_words import identify_difficult_words, sort_words
from simplify import simplify

DATA = Path("input/test_data.txt")
BACKUP = Path("input/test_data_original.txt")
REPLACEMENTS = Path("data/replacement_words.csv")
REPLACE_LIST = Path("input/replace.txt")
KEYWORDS = Path("input/keywords.txt")
BERT_OUTPUT = Path("output/test_data_bert.txt")
REPORT = Path("output/replacements.txt")
NO_MATCH = Path("input/no_replacement.txt")

def load_input():
    print("Enter text to simplify (press Enter twice to finish):")

    lines = []

    while True:
        line = input()

        if not line:
            break

        lines.append(line)

    text = "\n".join(lines)

    with open(DATA, "w", encoding="utf-8") as file:
        file.write(text)

    return text


def load_replacements(path):
    with open(path, newline="", encoding="utf-8") as file:
        return {row["phrase"].lower(): row["replacement"] for row in csv.DictReader(file)}

def write_lines(path, lines):
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

def delete_files():
    DATA.unlink(missing_ok=True)
    BACKUP.unlink(missing_ok=True)
    REPLACE_LIST.unlink(missing_ok=True)
    KEYWORDS.unlink(missing_ok=True)
    BERT_OUTPUT.unlink(missing_ok=True)
    REPORT.unlink(missing_ok=True)
    NO_MATCH.unlink(missing_ok=True)

def main():
    load_input()
    if not BACKUP.exists():
        shutil.copy(DATA, BACKUP)

    original = DATA.read_text(encoding="utf-8")
    replacements = load_replacements(REPLACEMENTS)
    simplified = simplify(original, replacements)
    DATA.write_text(simplified, encoding="utf-8")

    replace_words, flag_words = identify_difficult_words(simplified)
    replace_words = sort_words(replace_words)
    flag_words = sort_words(flag_words)
    write_lines(REPLACE_LIST, replace_words)
    write_lines(KEYWORDS, flag_words)

    output, report, no_match = replace_with_bert(simplified, set(replace_words))
    BERT_OUTPUT.write_text(output, encoding="utf-8")
    write_lines(REPORT, [f"{old} -> {new}\t{sentence}" for old, new, sentence in report])
    write_lines(NO_MATCH, no_match)

    print("ORIGINAL\n" + original.strip() + "\n")
    print("SIMPLIFIED\n" + simplified + "\n")
    print("REPLACE:", ", ".join(replace_words))
    print("KEEP/EXPLAIN:", ", ".join(flag_words) + "\n")
    print("BERT REPLACEMENTS")
    for old, new, _sentence in report:
        print(f"  {old} -> {new}")
    print("NO GOOD REPLACEMENT:", ", ".join(no_match) + "\n")
    print("FINAL\n" + output)

    delete_files()

if __name__ == "__main__":
    main()