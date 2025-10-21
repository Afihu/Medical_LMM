from transformers import pipeline
from PIL import Image
import torch

pipe = pipeline(
    "image-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda" # Specify "cuda" if you have a GPU, otherwise use "cpu"
)

try:
    image = Image.open('../res/extracted_images/page2_img1.jpeg')
except FileNotFoundError:
    print("Error: Image file not found. Please update the path.")
    exit()

prompt = "Analyze this medical image and generate a detailed report, summarizing all significant findings and a potential impression."

print("Generating report...")
output = pipe(
    images=image,
    text=prompt,
    max_new_tokens=1024,  # Adjust token length for detailed reports
)

# 5. Print the result
if output and output[0].get("generated_text"):
    summary = output[0]["generated_text"]
    print("\n--- Generated Summary ---")
    print(summary)
else:
    print("Model did not generate a summary.")