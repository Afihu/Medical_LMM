import fitz  # import  PyMuPDF library
import re

try:
    pdf_path = "source_materials/1---A-20-Year-Old-Woman-from-Sudan-With-Fever--_2022_Clinical-Cases-in-Tropi.pdf"
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


    # save to file
    output_text_file = "preprocessor/res/extracted_text.txt"
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(cleaned.lstrip())
        
    print(f"Successfully extracted text to '{output_text_file}'")

    doc.close()

except FileNotFoundError:
    print(f"Error: The file '{pdf_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

