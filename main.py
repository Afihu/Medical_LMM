from transformers import AutoModelForCausalLM, AutoTokenizer
from analysis import config_init
from peft import get_peft_model
import torch
import time

original_svd = torch.linalg.svd

def patched_svd(A, full_matrices=True, driver=None, *args, **kwargs):
    U, S, Vh = original_svd(A.float(), full_matrices=full_matrices, driver=driver, *args, **kwargs)
    return U.to(torch.bfloat16), S.to(torch.bfloat16), Vh.to(torch.bfloat16)

torch.linalg.svd = patched_svd

model_name = "Qwen/Qwen3-8B"

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map={"": "cpu"},
    low_cpu_mem_usage=True
)

config = config_init()

print("Begin SVD")
start_time = time.perf_counter()

model = get_peft_model(model, config) # Trigger SVD
model.print_trainable_parameters() # Verify trainable parameters

end_time = time.perf_counter()
execution_time = end_time - start_time

print(f"Elapsed time: {execution_time:.4f} seconds")

print("Success")