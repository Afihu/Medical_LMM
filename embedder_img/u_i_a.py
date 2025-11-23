from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch

def embed_img(image_path, caption):
    # the caption should be something like this:
    # caption = [
    #     "Some guy died"
    # ]
    # or it can be left blank and the image can be embedded on its own

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}") # <-- Add this line

    model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")

    img = [Image.open(image_path).convert("RGB")]

    inputs = processor(text=caption, images=img, padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = torch.softmax(logits_per_image, dim=1)
    
    return outputs
