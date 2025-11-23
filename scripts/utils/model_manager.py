"""
Model Manager (Singleton Pattern)
Manages model instances across pipelines (Embedding & Query).
Ensures only one instance of each model is loaded at any time.
Intelligently selects device (GPU/CPU) and optimizes memory usage.
"""

import torch
import gc


class ModelManager:
    """Singleton pattern for managing model instances across pipelines."""
    
    _instance = None
    _models = {}
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._device = cls._select_device()
        return cls._instance
    
    @staticmethod
    def _select_device():
        """
        Intelligently select device (GPU if available, CPU otherwise).
        Logs device information for debugging.
        """
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[INFO] GPU detected for model loading")
            print(f"  Device: {device_name}")
            print(f"  Total VRAM: {total_vram:.2f} GB")
            return "cuda"
        else:
            print(f"[INFO] No GPU detected. Models will run on CPU")
            print(f"  (To use GPU, ensure CUDA is installed and available)")
            return "cpu"
    
    @classmethod
    def get_device(cls):
        """Get the current device being used for models."""
        if cls._device is None:
            cls._device = cls._select_device()
        return cls._device
    
    @classmethod
    def get_text_generator(cls, model_name="NeuML/pubmedbert-base-embeddings", output_base_dir=None):
        """Get or create text embedding generator (singleton per model)."""
        from scripts.embedding_generation_module.generators.text_embedding_generator import TextEmbeddingGenerator
        
        key = f"text_{model_name}"
        if key not in cls._models:
            print(f"[INFO] Initializing text embedding generator: {model_name}")
            cls._models[key] = TextEmbeddingGenerator(model_name, output_base_dir)
        return cls._models[key]
    
    @classmethod
    def get_image_generator(cls, model_name="google/medsiglip-448", output_base_dir=None):
        """Get or create image embedding generator (singleton per model)."""
        from scripts.embedding_generation_module.generators.image_embedding_generator import ImageEmbeddingGenerator
        
        key = f"image_{model_name}"
        if key not in cls._models:
            print(f"[INFO] Initializing image embedding generator: {model_name}")
            cls._models[key] = ImageEmbeddingGenerator(model_name, output_base_dir)
        return cls._models[key]
    
    @classmethod
    def clear_models(cls):
        """
        Clear all cached models from memory (useful for memory management).
        Especially important when processing sequentially on memory-constrained systems.
        """
        if cls._models:
            model_count = len(cls._models)
            cls._models.clear()
            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[OK] {model_count} model(s) cleared from memory")
        else:
            print(f"[INFO] No models to clear")
