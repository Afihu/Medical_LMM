## How to Run the Fine-Tuned QPiSSA Medical Model

This model utilizes a **QPiSSA (Quantized Principal Singular values and Singular vectors Adaptation)** architecture. Unlike standard models, it is split into two distinct parts to optimize performance and memory usage:

1. `./Qwen-PiSSA-Residual-Base`: The main model weights (4-bit quantized residual).
2. `./Qwen-QPiSSA-Adapter-Final`: The fine-tuned adapter weights (with knowledge-distilled data).

**You must have both folders to run evaluation.**

### 1. Prerequisites

Ensure you have the required Python libraries installed. You need a GPU for 4-bit quantization support.

```bash
pip install torch transformers peft bitsandbytes accelerate
```

Run this on third-party hardware like Colab or Kaggle with GPU unless you're masochistic or swimming in money