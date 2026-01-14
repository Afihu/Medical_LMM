import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*20}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*20}\n")
    
    # Run the script and wait for it to finish
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}.")
        print("Aborting the rest of the pipeline to prevent corrupted training.")
        sys.exit(result.returncode)
    
    print(f"\n[SUCCESS] {script_name} finished successfully.")

if __name__ == "__main__":
    # 1. Run the CPU-based SVD Decomposition
    run_script("qdecompose.py")
    
    # 2. Check if the required residual base was actually saved
    if not os.path.exists("./Qwen-PiSSA-Residual-Base"):
        print("[ERROR] Residual base directory not found. Did Step 1 save correctly?")
        sys.exit(1)
        
    # 3. Run the GPU-based Training
    run_script("qtrainer.py")
    
    print("\n[COMPLETE] Full QPiSSA pipeline finished successfully.")