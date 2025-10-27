from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import time

start_time = time.perf_counter()

model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
image_path = "embedder_img/img/page2_img1.jpeg"
try:
    image = Image.open(image_path)
    print("Found")
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    print("Please update the 'image_path' variable to a valid file.")
    exit()

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert doctor."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image of a patient"},
            {"type": "image", "image": image}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

end_time = time.perf_counter()

duration = end_time - start_time
print(f"The program ran for: {duration:.4f} seconds")

# output = pipe(text=messages, max_new_tokens=200)
# print(output[0]["generated_text"][-1]["content"])