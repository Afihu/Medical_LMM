from svd_initializer import fast_config_init
from decomposer import decompose
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
import time 
import torch

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

#measuring convergence
class TimeTrackingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            elapsed = time.time() - self.start_time
            print(f"Step: {state.global_step} | Loss: {logs['loss']:.4f} | Elapsed: {elapsed:.2f}s")

# Initialize and retrieve tokenizer and model
svd_config = fast_config_init(256)
model, tokenizer = decompose(svd_config)

print("--- 3. Loading Saved PiSSA State ---")
# Overwrites the Base Model (injecting the residual) AND the Adapters (injecting principal components)
state_dict = torch.load("pissa_init_snapshot.pt", map_location="cpu")
model.load_state_dict(state_dict)

print("--- Model Loaded Successfully ---")

# --- DATA LOADING SECTION ---
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

# --- TRAINING SECTION ---
start = time.perf_counter()
print("--- Starting Training ---")
training_args = TrainingArguments(
    output_dir="./pissa-medical-finetune",
    per_device_train_batch_size=1,     
    gradient_accumulation_steps=4,     
    # gradient_checkpointing=True, # CRITICAL: Trades speed for massive VRAM savings       
    learning_rate=2e-5,                
    num_train_epochs=3,                
    save_steps=50,
    logging_steps=10,
    optim="paged_adamw_8bit",          
    bf16=True,                                              
    dataloader_pin_memory=False,    

    # DEVICE SETTINGS
    no_cuda=False,                      # Use GPU if possible
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[TimeTrackingCallback()]
)

trainer.train()

print("Saving Adapter...")
model.save_pretrained("./final_medical_adapter")
print("Done!")

end = time.perf_counter()
execution_time = end - start
print(f"Elapsed Training time: {execution_time:.4f} seconds")