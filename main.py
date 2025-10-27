from preprocessor import caption_extract, text_extract
from embedder_img import embedder

def main():
    print("Init texting external modules")

    pdf_path = "source_materials/5---A-4-Year-Old-Boy-from-Laos-With-a-Lesion-o_2022_Clinical-Cases-in-Tropic.pdf"

    pdf_texts = text_extract.extract_text(pdf_path)
    
    captions = caption_extract.extract_captions(pdf_texts)
    for caption in captions: 
        print(caption)


    embed = embedder.embed_img("embedder_img/img/page2_img1.jpeg", "Oral bleeding in Ebola virus disease. (Bausch, D.G., 2008. Viral hemorrhagic fevers. In: Schlossberg, D. (Ed.), Clinical Infectious Disease. Cambridge University Press, New York. Used with permission. Photo by Bausch, D.)")
    print(embed.image_embeds)


if __name__ == "__main__":
    main()
