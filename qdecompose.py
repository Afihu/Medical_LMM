import torch
import os
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

original_svd = torch.linalg.svd

def hybrid_svd(A, full_matrices=True, driver=None, *args, **kwargs):
    if torch.cuda.is_available():
        A_compute = A.to(device="cuda", dtype=torch.float32)
    else:
        A_compute = A.float()
    
    U, S, Vh = original_svd(A_compute, full_matrices=full_matrices, driver=driver, *args, **kwargs)

    # Move result back to CPU immediately to free VRAM
    return (
        U.to(device="cpu", dtype=torch.bfloat16), 
        S.to(device="cpu", dtype=torch.bfloat16), 
        Vh.to(device="cpu", dtype=torch.bfloat16)
    )

torch.linalg.svd = hybrid_svd

model_name = "Qwen/Qwen3-8B"
save_dir_base = "./Qwen-PiSSA-Residual-Base"  
save_dir_adapter = "./Qwen-PiSSA-Adapter"     

print(f"Loading {model_name} on CPU... (System RAM)")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto", 
    device_map={"": "cpu"},          
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Running PiSSA SVD...")
start_time = time.time()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=256,              
    lora_alpha=256,     
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    init_lora_weights="pissa" 
)

model = get_peft_model(model, peft_config)

print(f"SVD Complete in {time.time() - start_time:.2f} seconds.")

print("Saving Residual Base Model...")
model.base_model.model.save_pretrained(save_dir_base)
tokenizer.save_pretrained(save_dir_base)

print("Saving PiSSA Adapters...")
model.save_pretrained(save_dir_adapter)

config_path = os.path.join(save_dir_adapter, "adapter_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

config["init_lora_weights"] = False 
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("Decomposition Complete!")