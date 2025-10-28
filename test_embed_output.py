"""
test_embedder.py
----------------
Embeds all PDF images using MedSigLIP and saves them as a JSON file
for Qdrant upload. Each PDF may contain multiple images.

Output format:
[
  {
    "id": "<uuid>",                 # unique per image
    "vector": { "image_vector": [...] },
    "payload": {
        "case_id": "case_001",      # from filename prefix
        "image": "<base64 image data>"
    }
  },
  ...
]
"""

from preprocessor import caption_extract, text_extract, image_extract
from embedder_img import embedder
import glob
import os
import json
import base64
import io
import torch
import numpy as np
from PIL import Image
import uuid


def tensor_to_list(tensor):
    """Convert PyTorch tensor to a CPU list."""
    return tensor.detach().cpu().numpy().flatten().tolist()


def image_to_base64(image_array):
    """Convert NumPy image array to base64 string."""
    img = Image.fromarray(image_array).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_case_id(pdf_name):
    """Extract numeric prefix as case ID (e.g., '01_xray.pdf' → 'case_001')."""
    prefix = pdf_name.split("_")[0].split(".")[0]
    digits = "".join(filter(str.isdigit, prefix))
    if digits:
        return f"case_{int(digits):03d}"
    else:
        return f"case_{pdf_name.replace('.pdf', '')}"


def test_embedder():
    all_document_embeddings = [] 

    # === CONFIG ===
    pdf_directory = "D:/Bao/Document/_VGU/_Semester 5/Projects/ScienceDirect_articles_21Mar2025_16-17-58.017"
    output_json_path = os.path.join(os.getcwd(), "embedded_cases.json")

    pdf_paths = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    if not pdf_paths:
        print(f"Error: No PDF files found in '{pdf_directory}'.")
        exit() 

    print(f"Found {len(pdf_paths)} PDF file(s). Starting batch embedding...\n")

    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
        case_id = extract_case_id(pdf_name)

        try:
            print(f"\n--- Starting embedding for: {pdf_name} ---")

            # === Extract content ===
            image_arrays = image_extract.extract_image(pdf_path)
            document_text = text_extract.extract_text(pdf_path)
            caption_list = caption_extract.extract_captions(document_text)

            if not image_arrays:
                print(f"Skipping {pdf_name}: No images found.")
                continue

            # Ensure captions match image count
            if len(caption_list) < len(image_arrays):
                caption_list += [""] * (len(image_arrays) - len(caption_list))

            # === Embed images ===
            embeddings_for_pdf = embedder.embed_imgs(image_arrays, caption_list, pdf_name)

            for emb, img_arr in zip(embeddings_for_pdf, image_arrays):
                image_vector = tensor_to_list(emb["image_embedding"])
                image_b64 = image_to_base64(img_arr)

                point = {
                    "id": str(uuid.uuid4()),  # unique id for Qdrant
                    "vector": { "image_vector": image_vector },
                    "payload": {
                        "case_id": case_id,
                        "image": image_b64
                    }
                }
                all_document_embeddings.append(point)

            print(f"Embedded {len(embeddings_for_pdf)} image(s) from '{pdf_name}'.")

        except Exception as e:
            print(f"Error processing {pdf_name}: {e}")
            continue

    # === Save output JSON ===
    if all_document_embeddings:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(all_document_embeddings, f, indent=4, ensure_ascii=False)

        print("\n=======================================================")
        print(f"Batch complete. Total embedded images: {len(all_document_embeddings)}")
        print(f"Saved to: {output_json_path}")
        print("=======================================================")
    else:
        print("No valid embeddings generated — nothing saved.")


if __name__ == "__main__":
    test_embedder()
