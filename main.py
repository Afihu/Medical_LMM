import subprocess
import sys
import os

BASE_DIR = "./Qwen-PiSSA-Residual-Base"
ADAPTER_DIR = "./Qwen-PiSSA-Adapter"

def decomposition_check():
    base_exists = os.path.exists(os.path.join(BASE_DIR, "config.json"))
    adapter_exists = os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json"))
    return base_exists and adapter_exists

def run_script(script_name):
    print(f"\n{'='*20}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*20}\n")
    
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}.")
        print("Aborting the rest of the pipeline to prevent corrupted training.")
        sys.exit(result.returncode)
    
    print(f"\n[SUCCESS] {script_name} finished successfully.")

if __name__ == "__main__":
    # Check if can skip SVD
    if decomposition_check():
        print(f"\n[INFO] Found existing PiSSA files in '{BASE_DIR}'.")
        print("[INFO] Skipping SVD Decomposition...")
    else:
        print("[INFO] No existing PiSSA files found. Starting SVD...")
        run_script("decompose_qpissa.py")
    
    # Training
    run_script("qtrainer.py")
    
    print("\n[COMPLETE] Pipeline finished successfully.")