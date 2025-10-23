import fitz  # import  PyMuPDF library
import re

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        
        print(f"Opened '{pdf_path}', which has {doc.page_count} pages.")

        full_text = ""
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)  
            text = page.get_text("text")   # Extract text from the page
            full_text += text + "\n"
        
        # cleaning
        phrase = "Further Reading"
        parts = full_text.split(phrase)
        cleaned = parts[0]
        cleaned = re.sub(r'• Figure|Fig\.\s\d+\.\d+\s*', '', cleaned)

        doc.close()

        return full_text

    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

