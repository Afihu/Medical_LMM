from preprocessor import caption_extract

def main():
    print("Init texting external modules")

    pdf_path = "source_materials/3---A-26-Year-Old-Woman-from-Malawi-with-Headache-_2022_Clinical-Cases-in-Tr.pdf"

    caption_extract.extract_captions(pdf_path)

    


if __name__ == "__main__":
    main()
