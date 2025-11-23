import fitz  # PyMuPDF
import os
import re
import unicodedata


# --- Function to clean text from newlines ---
def clean_text(text):
    if text is None:
        return ""
    # Replace '\n' with space if not preceded by '-'
    text = re.sub(r'(?<!-)\n', ' ', text)
    # Remove '\n' if preceded by '-'
    text = re.sub(r'-\n', '', text)
    # Remove carriage returns
    text = re.sub(r'\r', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# --- Helper: normalize and pre-clean text before regex ---
def normalize_text(t):
    if t is None:
        return ""
    t = unicodedata.normalize("NFKC", t)
    # Replace mis-encoded characters common in author names
    t = t.replace("€", "Ü").replace("Â", "")
    # Remove weird spacing artifacts
    t = re.sub(r"[ \t]+", " ", t)
    return t


# --- 1. Function: Extract relevant sections from each PDF ---
def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")

    # Normalize and lightly flatten text before regex
    text = normalize_text(text)

    # Flatten hyphenated and broken lines for better matching
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # --- Extract Title  ---
    if "A 14-Year-Old Girl in the Solomon Islands With a Non-Healing Leg Ulcer" in text:
        title = "A 14-Year-Old Girl in the Solomon Islands With a Non-Healing Leg Ulcer"
    else:
        title = None
        pattern_multiline = (
            r"^\s*(?:Case\s*\d+:|\d+)\s*\n"
            r"((?:[^\n]*\n)+?)"
            r"(?=\n?[A-ZÁÉÍÓÚÑÜ\s,.\-&]{3,}\n)"
        )
        match_multiline = re.search(pattern_multiline, text, re.M)
        if match_multiline:
            title = re.sub(r"\s*\n\s*", " ", match_multiline.group(1).strip())

        if not title:
            match_case_line = re.search(
                r"(?:Case\s*\d+:|\d+)\s*([A-Z][^\n]+?)(?=\s+(?:MICHAEL|Clinical Presentation|History))",
                text, re.I
            )
            if match_case_line:
                title = match_case_line.group(1).strip()

        if not title:
            match_fallback = re.search(
                r"^\s*(?:Case\s*\d+:|\d+)\s*\n\s*([A-Z][^\n]{10,}?)\s+(?=MICHAEL|Clinical Presentation)",
                text, re.M
            )
            if match_fallback:
                title = match_fallback.group(1).strip()

        if title:
            title = re.sub(r"\s+", " ", title).strip()


    section_boundaries = (
        "Clinical Presentation|History|Clinical Findings|Clinical Examination|Examination|Examination findings|Physical Examination|"
        "Questions|Discussion|Laboratory Results|Investigations|Further Investigations|"
        "Laboratory Findings|Laboratory Results and Imaging|Abdominal Ultrasound|"
        "The Case Continued|SUMMARY BOX|Answer to Question|Diagnosis|Treatment"
    )

    # --- Extract "History" section ---
    history_pattern = rf"History\s*\n(.*?)(?=\n(?:{section_boundaries})\b)"
    history_match = re.search(history_pattern, text, re.S | re.I)
    history = history_match.group(1).strip() if history_match else None

    # --- Extract "Clinical Findings" section ---
    findings_pattern = rf"(?:Clinical Findings|Clinical Examination|Examination|Examination findings|Physical Examination)\s*\n(.*?)(?=\n(?:{section_boundaries})\b)"
    findings_match = re.search(findings_pattern, text, re.S | re.I)
    findings = findings_match.group(1).strip() if findings_match else None

    # --- Extract "Discussion" section ---
    discussion_pattern = rf"Discussion\s*\n(.*?)(?=\n(?:Answer to Question 1)|What are the priorities for management?|Answer Question 1\b)"
    discussion_match = re.search(discussion_pattern, text, re.S | re.I)
    discussion = discussion_match.group(1).strip() if discussion_match else None

    # --- Extract "SUMMARY BOX" (diagnosis line) ---
    summary_box_match = re.search(r"SUMMARY BOX\s*\n([^\n]*)", text)
    summary_box_first_line = summary_box_match.group(1).strip() if summary_box_match else None

    # Apply cleaning to extracted sections
    return {
        "Title": clean_text(title),
        "History": clean_text(history),
        "Clinical Findings": clean_text(findings),
        "Discussion": clean_text(discussion),
        "Summary Box First Line": clean_text(summary_box_first_line),
    }


# --- Backward compatibility function ---
def extract_text(pdf_path):
    """Extract text sections from PDF. Returns structured sections dictionary."""
    return extract_sections(pdf_path)


# Helper function for debugging
def txt_print(texts):
    """Helper function to write extracted text to file for debugging."""
    print(f"Writing content to out.txt")
    file_name = "out.txt"
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(texts)
        print(f"Successfully wrote the text to '{file_name}'")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
