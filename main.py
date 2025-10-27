from preprocessor import caption_extract, text_extract

def main():
    print("Init texting external modules")

    pdf_path = "source_materials/5---A-4-Year-Old-Boy-from-Laos-With-a-Lesion-o_2022_Clinical-Cases-in-Tropic.pdf"

    pdf_texts = text_extract.extract_text(pdf_path)
    captions = caption_extract.extract_captions(pdf_texts)
    
    for caption in captions: 
        print(caption)


if __name__ == "__main__":
    main()
