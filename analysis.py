from peft import LoraConfig, TaskType

def config_init():
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=256,              # Rank: Critical for PiSSA (determines SVD components)
        lora_alpha=256,     # Set alpha = r for PiSSA
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention layers

        # --- THIS IS THE KEY ---
        init_lora_weights="pissa" 
    )

    return peft_config

if __name__ == "__main__":
    config_init()
    print("Success")