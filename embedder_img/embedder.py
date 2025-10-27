# import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch

# # Debugging import
# from preprocessor import text_extract
# import numpy as np

def embed_img(image_path, texts):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}") # <-- Add this line

    model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")

    img = [Image.open(image_path).convert("RGB")]

    inputs = processor(text=texts, images=img, padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = torch.softmax(logits_per_image, dim=1)
    
    return outputs


### Embedder function of interest
### The main embedding mechanism will go through each PDF to extract images and captions, store in arrays, 
# then feed into embedder
def embed_imgs(imagelist, caption_list, pdf_name):
    if len(imagelist) != len(caption_list):
        raise ValueError(
            f"EmbeddingError: Mismatch in '{pdf_name}'\n"
            f"Found {len(imagelist)} images but {len(caption_list)} captions."
        )

    print(f"Check passed for '{pdf_name}': {len(imagelist)} images and {len(caption_list)} captions.")
    
    # --- Load Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")

    embeddings_list = []

    # embed each image against its matching caption
    for i, (image_array, caption_text) in enumerate(zip(imagelist, caption_list)):
        img = [Image.fromarray(image_array).convert("RGB")]
        
        inputs = processor(text=[caption_text], images=img, padding="max_length", return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings_list.append({
            "pair_index": i,
            "image_embedding": outputs.image_embeds,
            "text_embedding": outputs.text_embeds,
            "caption": caption_text
        })
        print(f"  Embedded pair {i+1} from '{pdf_name}'")

    print(f"Successfully embedded {len(embeddings_list)} pairs from '{pdf_name}'.")
    return embeddings_list


# #### debugging script
# python_num = outputs.image_embeds
# cpu_tensor = python_num.cpu()

# numpy_array = cpu_tensor.numpy()
# np.set_printoptions(threshold=np.inf, linewidth=200)
# res = str(numpy_array)

# text_extract.txt_print(res)

# # Get the image and text embeddings
# print(f"image embeddings: {outputs.image_embeds}")
# print(f"text embeddings: {outputs.text_embeds}")
