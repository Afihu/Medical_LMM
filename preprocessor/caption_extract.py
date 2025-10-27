import re

def extract_captions(text_doc):

    captions  = []
    patterns = r"^(•\s*)?(fig|figure|table)\.?\s*\d"

    print(f"--- Searching for captions ---")

    for line in text_doc.splitlines():
        clean_line = line.strip()
        if re.search(patterns, clean_line, re.IGNORECASE):
            captions.append(clean_line)

    return captions