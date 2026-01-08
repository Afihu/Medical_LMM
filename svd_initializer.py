from peft import LoraConfig, TaskType

def config_init(rank):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=rank,              # Rank: Critical for PiSSA (determines SVD components)
        lora_alpha=rank,     # Set alpha = r for PiSSA
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention layers

        init_lora_weights="pissa" 
    )

    return peft_config

# Only use when SVD has been run before
def fast_config_init(rank):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        init_lora_weights=False  # Skips SVD
    )

if __name__ == "__main__":
    config_init()
    print("Success")