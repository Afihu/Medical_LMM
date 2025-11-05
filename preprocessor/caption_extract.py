import re
import yake
# from preprocessor import text_extract

def extract_captions(text_doc):
    captions = []
    
    start_pattern = r"^(•\s*)?(fig|figure)\.?\s*\d"
    
    end_pattern = re.compile(r"(\n){2,}")
    
    print(f"--- Searching for captions ---")

    matches = list(re.finditer(start_pattern, text_doc, re.IGNORECASE | re.MULTILINE))
    
    if not matches:
        print("No captions found.")
        return []

    for match in matches:
        start_index = match.start()
        
        end_match = end_pattern.search(text_doc, pos=start_index)
        
        if end_match:
            end_index = end_match.start()
            raw_caption = text_doc[start_index:end_index]
        else:
            raw_caption = text_doc[start_index:]
            
        clean_caption = re.sub(r'\s+', ' ', raw_caption).strip()

        if not clean_caption:
            continue
        
        table_detect_pattern = r"\sTABLE\s*\d"
        table_match = re.search(table_detect_pattern, clean_caption, re.IGNORECASE)
        
        if table_match:
            clean_caption = clean_caption[:table_match.start()].strip()
            
        if clean_caption:
            captions.append(clean_caption)

    return captions

# Run pip install yake before running this code
def extract_keyword(text_doc):
    kw_extractor = yake.KeywordExtractor(
        lan = "en",
        n=5,
        dedupLim=0.9,
        top=15
    )

    keywords = kw_extractor.extract_keywords(text_doc)

    print("keywords: \n")
    count = 0
    for kw, score in keywords:
        print(kw)



# if __name__ == "__main__":
#     text_doc = text_extract.extract_text("source_materials/2---A-7-Year-Old-Girl-from-Peru-With-a-Chron_2022_Clinical-Cases-in-Tropical.pdf")
#     text_extract.txt_print(text_doc)
#     captions = extract_captions(text_doc=text_doc)
#     print(len(captions))
#     for caption in captions:
#         print(caption, "\n\n")

if __name__ == "__main__":
    text = """
        Ms. Anya Sharma, a 28-year-old software engineer, presented to the clinic with an acute onset of high fever, severe retro-orbital pain, and debilitating joint and muscle pain, often referred to as "breakbone fever." She returned four days ago from a week-long vacation in Southeast Asia where she reported frequent daytime mosquito bites. Initial blood work reveals leukopenia and progressive thrombocytopenia (platelets currently 90,000/μL), raising strong clinical suspicion for Dengue Fever. The patient is currently being monitored for any warning signs of severe plasma leakage or bleeding complications, requiring close management of fluid balance.
    """
    extract_keyword(text)