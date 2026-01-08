from transformers import AutoModelForCausalLM, AutoTokenizer
from svd_initializer import config_init
from peft import get_peft_model
import torch
import time
from safetensors.torch import save_file

original_svd = torch.linalg.svd

# "hybrid" for CPU and GPU incorporation, balance between memory efficiency and processing time
def hybrid_svd(A, full_matrices=True, driver=None, *args, **kwargs):
    if torch.cuda.is_available():
        A_compute = A.to(device="cuda", dtype=torch.float32)
    else:
        A_compute = A.float()
    
    U, S, Vh = original_svd(A_compute, full_matrices=full_matrices, driver=driver, *args, **kwargs)

    # Cast back to bfloat16 and move result back to CPU to store in the model
    return (
        U.to(device="cpu", dtype=torch.bfloat16), 
        S.to(device="cpu", dtype=torch.bfloat16), 
        Vh.to(device="cpu", dtype=torch.bfloat16)
    )

torch.linalg.svd = hybrid_svd

def decompose(svd_config):
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load Tokenizer and Model
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )

    config = svd_config

    if torch.cuda.is_available():
        print("CUDA")
    else:
        print("CPU")

    print("Begin SVD")
    start_time = time.perf_counter()

    model = get_peft_model(model, config) # Trigger SVD depending on donfig
    model.print_trainable_parameters() # Verify trainable parameters

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Elapsed SVD time: {execution_time:.4f} seconds")

    # Saves the entire state_dict (Base Model Weights + Adapter Weights)
    # This file will be large (~15GB), use with caution.
    save_file(model.state_dict(), "pissa_init_snapshot.safetensors")

    print("Success")

    return model, tokenizer