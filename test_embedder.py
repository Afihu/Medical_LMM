from preprocessor import caption_extract, text_extract, image_extract
from embedder_img import embedder
import glob
import os

def test_embedder():
    all_document_embeddings = [] 
    
    pdf_directory = "source_materials"

    pdf_paths = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    if not pdf_paths:
        print(f"Error: No PDF files found in the directory '{pdf_directory}'.")
        exit() 

    print(f"Found {len(pdf_paths)} PDF file(s). Starting batch processing...")

    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
        
        try:
            print(f"\n--- Starting Full Pipeline for: {pdf_name} ---")
            
            image_arrays = image_extract.extract_image(pdf_path)
            
            document_text = text_extract.extract_text(pdf_path)
            
            caption_list = caption_extract.extract_captions(document_text)

            if not image_arrays or not caption_list:
                print(f"Skipping {pdf_name}: Images={len(image_arrays)}, Captions={len(caption_list)}. Cannot proceed with embedding.")
                continue

            embeddings_for_pdf = embedder.embed_imgs(image_arrays, caption_list, pdf_name)
            
            all_document_embeddings.extend(embeddings_for_pdf)
            
            print(f"Successfully processed and embedded {len(embeddings_for_pdf)} pairs from {pdf_name}.")

        except ValueError as e:
            print(f"\nFATAL ERROR: {e}")
            print(f"Processing halted for {pdf_name}. Skipping to next document.")
            continue
        except Exception as e:
            print(f"\nAN UNEXPECTED ERROR occurred while processing {pdf_name}: {e}")
            continue

    print("\n=======================================================")
    print(f"BATCH PROCESS COMPLETE. Total embedded pairs: {len(all_document_embeddings)}")
    print("=======================================================")


if __name__ == "__main__":
    test_embedder()
