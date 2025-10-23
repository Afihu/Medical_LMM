import fitz  # import  PyMuPDF library
import re

def extract_captions(pdf_path):
    try:
        doc = fitz.open(pdf_path)

        print(f"--- Searching for captions ---")
        found_captions = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")

            # Filter for text blocks (type == 0)
            text_blocks = [b for b in blocks if b[6] == 0]

            for block in text_blocks:
                text = block[4].strip()
                if re.match(r"(•\s*)?(fig|figure|table)\.?\s*\d", text, re.IGNORECASE):
                
                    # Found one! Clean up newlines for printing.
                    clean_caption = text.replace('\n', ' ')
                    print(f"  [Found on Page {page_num + 1}]: {clean_caption}")
                    found_captions.append(clean_caption)

        if not found_captions:
            print("No captions were found.")
            
        doc.close()

    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")