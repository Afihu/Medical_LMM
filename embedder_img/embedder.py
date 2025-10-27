# import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch

# Debugging import
from preprocessor import text_extract
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}") # <-- Add this line

model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

image_path = "embedder_img/img/page2_img1.jpeg"

img = [Image.open(image_path).convert("RGB")]

texts = [
    "Oral bleeding in Ebola virus disease. (Bausch, D.G., 2008. Viral hemorrhagic fevers. In: Schlossberg, D. (Ed.), Clinical Infectious Disease. Cambridge University Press, New York. Used with permission. Photo by Bausch, D.)"
]

inputs = processor(text=texts, images=img, padding="max_length", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.softmax(logits_per_image, dim=1)


# #### debugging
# python_num = outputs.image_embeds
# cpu_tensor = python_num.cpu()

# numpy_array = cpu_tensor.numpy()
# np.set_printoptions(threshold=np.inf, linewidth=200)
# res = str(numpy_array)

# text_extract.txt_print(res)

# # Get the image and text embeddings
# print(f"image embeddings: {outputs.image_embeds}")
# print(f"text embeddings: {outputs.text_embeds}")
