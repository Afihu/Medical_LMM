from preprocessor import caption_extract, text_extract

def main():
    print("Init texting external modules")

    pdf_path = "source_materials/2---A-7-Year-Old-Girl-from-Peru-With-a-Chron_2022_Clinical-Cases-in-Tropical.pdf"

    pdf_texts = text_extract.extract_text(pdf_path)
    captions = caption_extract.extract_captions(pdf_texts)
    
    for caption in captions: 
        print(caption)

    


if __name__ == "__main__":
    main()
