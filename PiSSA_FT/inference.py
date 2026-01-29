import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# --- Configuration ---
model_id = "./princeps/model/Qwen-Base"
adapter_id = "./princeps/inference-ready/Qwen-QPiSSA-Adapter-FT"
input_file = "val-data.json"
output_file = "val-res.json"

#  8-bit
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)


print("Loading base")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

print("Plugging in the adapter")
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()


with open(input_file, "r") as f:
    val_data = json.load(f)

results = []

print(f"Starting inference on {len(val_data)} cases...")

for case in tqdm(val_data):
    case_text = case.get("prompt", "")
    prompt = (
        f"Analyze the clinical presentation and provide a diagnosis.\n\n"
        f"Patient Case:\n{case_text}\n\n"
        f"Rationale:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,  
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    results.append({
        "id": case.get("id"),
        "input_case": case_text,
        "expected_diagnosis": case.get("diagnosis") or case.get("expected_diagnosis"),
        "model_output": generated_text.strip()
    })

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Inference complete! Results saved to {output_file}")