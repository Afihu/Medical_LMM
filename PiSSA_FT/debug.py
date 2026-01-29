import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import gc

# --- Configuration ---
residual_base_path = "./princeps/model/Qwen-Base"
adapter_path = "./princeps/model/Qwen-QPiSSA-Adapter"
output_dir = "./princeps/inference-ready/Qwen-QPiSSA-Adapter-FT-experior"

data_file = "augmented-dataset.json"
validation_data = "val-data.json"

# --- 1. Load Data & Create Validation Split ---
print("--- Loading Data ---")
dataset = load_dataset("json", data_files=data_file, split="train")
eval_dataset = load_dataset("json", data_files=validation_data, split="train")

train_dataset = dataset.rename_column("prompt", "case_text")
eval_dataset = eval_dataset.rename_column("prompt", "case_text")

print(f"Training on {len(train_dataset)} samples")
print(f"Validating on {len(eval_dataset)} samples")

# --- 2. Load Residual Base (4-bit) ---
print("--- Loading Residual Base (4-bit) ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    residual_base_path,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)

model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(residual_base_path)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 3. Load PiSSA Adapters ---
print("--- Loading PiSSA Adapters ---")
model = PeftModel.from_pretrained(
    model,
    adapter_path,
    is_trainable=True
)

model.print_trainable_parameters()

# --- 4. Training Arguments ---
print("--- Starting QPiSSA Training ---")

training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=12,
    logging_steps=5,
    optim="paged_adamw_32bit",
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,

    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=60,
    save_steps=60,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", # Watch the validation loss
    greater_is_better=False,

    weight_decay=0.2,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    max_length=1024,
    dataset_text_field="text",
    packing=False,
    report_to="none"
)

# --- 5. Pre-Processing & Masking Logic (Replaces Formatter) ---
def process_and_mask(sample):
    prompt_text = (
        f"Analyze the clinical presentation and provide a diagnosis.\n\n"
        f"Patient Case:\n{sample['case_text']}\n\n"
    )

    completion_text = (
        f"Diagnosis:\n{sample['diagnosis']}"
        f"{tokenizer.eos_token}"
    )

    # add_special_tokens=True adds BOS to the prompt
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True).input_ids
    completion_ids = tokenizer(completion_text, add_special_tokens=False).input_ids

    # Concatenate
    input_ids = prompt_ids + completion_ids

    # Create Labels: -100 masks the loss for prompt tokens
    prompt_mask = [-100] * len(prompt_ids)
    labels = prompt_mask + completion_ids

    # Optional: Truncate to max_length
    max_len = training_args.max_length
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels
    }

print("--- Pre-processing and Masking Data ---")
train_dataset = train_dataset.map(process_and_mask, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(process_and_mask, remove_columns=eval_dataset.column_names)

# --- 6. Initialize SFTTrainer ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),

    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

print("--- Saving Final Model ---")
trainer.save_model(output_dir)

history = trainer.state.log_history

print("Done Training!")

# --- 8. Plotting ---
train_steps = [x['step'] for x in history if 'loss' in x]
train_loss = [x['loss'] for x in history if 'loss' in x]

eval_steps = [x['step'] for x in history if 'eval_loss' in x]
eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label='Training Loss', color='blue')
if eval_loss:
    plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red', linestyle='--')

plt.title('Training vs Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save plot
plt.savefig(f"{output_dir}/loss_curve.png")
print(f"Graph saved to {output_dir}/loss_curve.png")