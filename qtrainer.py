import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

# --- CONFIGURATION ---
# Load the LOCAL folders we created in Step 1
residual_base_path = "./Qwen-PiSSA-Residual-Base"
adapter_path = "./Qwen-PiSSA-Adapter"
output_dir = "./qwen-qpissa-final"

# 1. Load the RESIDUAL Base Model in 4-bit
# This is "QPiSSA": The base is quantized residual, adapters are high precision.
print("--- Loading Residual Base (4-bit) ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 
)

model = AutoModelForCausalLM.from_pretrained(
    residual_base_path,   # Load from our local folder
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False 
)
tokenizer = AutoTokenizer.from_pretrained(residual_base_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Attach the Pre-Calculated PiSSA Adapters
print("--- Loading PiSSA Adapters ---")
# We use PeftModel.from_pretrained to attach existing adapters to the base
model = PeftModel.from_pretrained(
    model, 
    adapter_path, 
    is_trainable=True # Ensure we can train them
)

model.print_trainable_parameters()
# You should see ~30-50M parameters trainable (the adapters).
# The billions of base parameters are frozen in 4-bit.

# --- 3. Data Loading ---
print("--- Loading Data ---")
dataset = load_dataset("json", data_files="data.json", split="train")

def format_medical_case(sample):
    formatted_text = (
        f"Analyze the clinical presentation and provide a diagnosis.\n\n"
        f"Patient Case:\n{sample['prompt']}\n\n"
        f"Diagnosis:\n{sample['diagnosis']}"
        f"{tokenizer.eos_token}" 
    )
    return {"text": formatted_text}

dataset = dataset.map(format_medical_case)
tokenized_datasets = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True)

# --- 4. Training ---
print("--- Starting QPiSSA Training ---")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,              
    num_train_epochs=3,
    logging_steps=10,
    optim="paged_adamw_32bit",       
    save_strategy="steps",
    save_steps=50,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,     
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

print("--- Saving Final Model ---")
model.save_pretrained(output_dir)
print("Done!")