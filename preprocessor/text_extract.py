import fitz  # import  PyMuPDF library
import re

def txt_print(texts):

    print(f"Writing content to out.txt")

    file_name = "out.txt"
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            # 2. Write the text to the file
            file.write(texts)
        
        print(f"Successfully wrote the text to '{file_name}'")

    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

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

        doc.close()

        return full_text

    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

