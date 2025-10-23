import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tensorflow.image import resize as tf_resize
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

image_path = "embedder_img/img/page2_img1.jpeg"

img = Image.open(image_path)

def resize(image):
    return Image.fromarray(
        tf_resize(
            images=image, size=[448, 448], method='bilinear', antialias=False
        ).numpy().astype(np.uint8)
    )


resized_imgs = [resize(img)]

texts = [
    "Oral bleeding in Ebola virus disease. (Bausch, D.G., 2008. Viral hemorrhagic fevers. In: Schlossberg, D. (Ed.), Clinical Infectious Disease. Cambridge University Press, New York. Used with permission. Photo by Bausch, D.)",
]

inputs = processor(text=texts, images=resized_imgs, padding="max_length", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.softmax(logits_per_image, dim=1)

for n_img, img in enumerate(imgs):
    display(img)  # Note this is an IPython function that will only work in a Jupyter notebook environment
    for i, label in enumerate(texts):
        print(f"{probs[n_img][i]:.2%} that image is '{label}'")

# Get the image and text embeddings
print(f"image embeddings: {outputs.image_embeds}")
print(f"text embeddings: {outputs.text_embeds}")
