"""
Embedding Orchestrator (Embedding Pipeline)
Orchestrates the generation of both text and image embeddings from extracted data.
Handles model loading and manages where embeddings are staged.
"""

import os
import sys
from pathlib import Path
import torch

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from scripts.config.embedding_config import CONFIG, PATHS
from scripts.embedding_generation_module.generators.text_embedding_generator import TextEmbeddingGenerator
from scripts.embedding_generation_module.generators.image_embedding_generator import ImageEmbeddingGenerator


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
        key = f"text_{model_name}"
        if key not in cls._models:
            print(f"[INFO] Initializing text embedding generator: {model_name}")
            cls._models[key] = TextEmbeddingGenerator(model_name, output_base_dir)
        return cls._models[key]
    
    @classmethod
    def get_image_generator(cls, model_name="google/medsiglip-448", output_base_dir=None):
        """Get or create image embedding generator (singleton per model)."""
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
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[OK] {model_count} model(s) cleared from memory")
        else:
            print(f"[INFO] No models to clear")


class EmbeddingOrchestrator:
    """
    Orchestrates embedding generation for the Embedding Pipeline.
    Handles both text and image embeddings, staging them to staged_embeddings/.
    """
    
    def __init__(self, output_base_dir=None, staging_mode="production"):
        """
        Initialize the embedding orchestrator.
        
        Args:
            output_base_dir: Base directory for outputs (default: project root)
            staging_mode: "production" (staged_embeddings/) or "temp" (temp_query_data/)
        """
        self.output_base = output_base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        self.staging_mode = staging_mode
        
        # Determine output directories based on staging mode
        if staging_mode == "production":
            self.text_output_dir = os.path.join(self.output_base, PATHS["staged_embeddings"]["text"])
            self.image_output_dir = os.path.join(self.output_base, PATHS["staged_embeddings"]["images"])
        elif staging_mode == "temp":
            self.text_output_dir = os.path.join(self.output_base, "temp_query_data", "text_embeddings")
            self.image_output_dir = os.path.join(self.output_base, "temp_query_data", "image_embeddings")
        else:
            raise ValueError(f"Unknown staging mode: {staging_mode}")
        
        # Ensure output directories exist
        os.makedirs(self.text_output_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)
        
        print(f"[INFO] Embedding Orchestrator initialized")
        print(f"  Staging Mode: {staging_mode}")
        print(f"  Text Output Directory: {self.text_output_dir}")
        print(f"  Image Output Directory: {self.image_output_dir}")
    
    def generate_all_embeddings(self):
        """
        Generate both text and image embeddings from extracted data.
        Uses sequential processing: text → clear memory → images.
        This optimizes memory usage, especially on CPU or memory-constrained systems.
        
        Returns:
            dict: Summary of embedding generation results
        """
        print(f"\n{'='*60}")
        print("EMBEDDING ORCHESTRATION STARTED")
        print(f"{'='*60}\n")
        
        results = {
            "text": [],
            "images": [],
            "summary": {
                "text_successful": 0,
                "image_successful": 0,
                "total_successful": 0
            }
        }
        
        # PHASE 1: Generate text embeddings
        print("[PHASE 1/2] Generating text embeddings...")
        text_generator = ModelManager.get_text_generator()
        text_results = self._generate_text_embeddings(text_generator)
        results["text"] = text_results
        results["summary"]["text_successful"] = sum(1 for r in text_results if r["success"])
        
        # CLEANUP: Free text model from memory before loading image model
        print("\n[MEMORY CLEANUP] Releasing text embedding model from memory...")
        ModelManager.clear_models()
        print("[OK] Text model unloaded from RAM/VRAM\n")
        
        # PHASE 2: Generate image embeddings
        print("[PHASE 2/2] Generating image embeddings...")
        image_generator = ModelManager.get_image_generator()
        image_results = self._generate_image_embeddings(image_generator)
        results["images"] = image_results
        results["summary"]["image_successful"] = sum(1 for r in image_results if r["success"])
        
        # Final cleanup
        print("\n[MEMORY CLEANUP] Releasing image embedding model from memory...")
        ModelManager.clear_models()
        print("[OK] Image model unloaded from RAM/VRAM\n")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _generate_text_embeddings(self, text_generator):
        """Generate text embeddings for all extracted text files."""
        input_dir = os.path.join(self.output_base, PATHS["extracted_data"]["text"])
        
        if not os.path.exists(input_dir):
            print(f"  [WARN] Text input directory not found: {input_dir}")
            return []
        
        text_files = sorted([f for f in os.listdir(input_dir) if f.startswith("case_") and f.endswith("_text.json")])
        
        if not text_files:
            print(f"  [WARN] No text files found in {input_dir}")
            return []
        
        results = []
        for text_file in text_files:
            case_id = int(text_file.split("_")[1])
            text_path = os.path.join(input_dir, text_file)
            
            print(f"  Processing: {text_file}")
            # Save directly to self.text_output_dir
            try:
                import json
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_sections = json.load(f)
                
                embedding = text_generator.generate_embedding(text_sections)
                if embedding is None:
                    result = {"success": False, "error": "No embedding generated"}
                else:
                    output_filename = PATHS["naming_conventions"]["text_embedding"].format(case_id=case_id) if "naming_conventions" in PATHS else f"case_{case_id:03d}_text_embedding.npy"
                    output_path = os.path.join(self.text_output_dir, output_filename)
                    
                    import numpy as np
                    np.save(output_path, embedding)
                    result = {"success": True, "path": output_path, "shape": embedding.shape}
                    print(f"  [OK] Saved to {output_filename}")
            except Exception as e:
                result = {"success": False, "error": str(e)}
                print(f"  [ERROR] {str(e)}")
            
            result["case_id"] = case_id
            result["input_file"] = text_file
            results.append(result)
        
        return results
    
    def _generate_image_embeddings(self, image_generator):
        """Generate image embeddings for all extracted image files."""
        image_dir = os.path.join(self.output_base, PATHS["extracted_data"]["images"])
        caption_dir = os.path.join(self.output_base, PATHS["extracted_data"]["captions"])
        
        if not os.path.exists(image_dir):
            print(f"  [WARN] Image input directory not found: {image_dir}")
            return []
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.startswith("case_") and f.endswith(".png")])
        
        if not image_files:
            print(f"  [WARN] No image files found in {image_dir}")
            return []
        
        results = []
        for image_file in image_files:
            # Extract case_id and image_id from filename
            parts = image_file.replace(".png", "").split("_")
            case_id = int(parts[1])
            image_id = int(parts[3])
            
            image_path = os.path.join(image_dir, image_file)
            
            # Load caption for this image
            import json
            caption_text = None
            caption_file = f"case_{case_id:03d}_captions.json"
            caption_path = os.path.join(caption_dir, caption_file)
            
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption_data = json.load(f)
                        if caption_data.get("captions"):
                            for cap in caption_data["captions"]:
                                if cap.get("id") == image_id:
                                    caption_text = cap.get("text", "")
                                    break
                except Exception as e:
                    print(f"  [WARN] Could not load caption: {e}")
            
            if not caption_text:
                caption_text = f"Image {image_id} from case {case_id}"
            
            print(f"  Processing: {image_file}")
            
            # Generate and save embedding
            try:
                embedding = image_generator.generate_embedding(image_path, caption_text)
                if embedding is None:
                    result = {"success": False, "error": "No embedding generated"}
                else:
                    output_filename = f"case_{case_id:03d}_image_{image_id:03d}_embedding.npy"
                    output_path = os.path.join(self.image_output_dir, output_filename)
                    
                    import numpy as np
                    np.save(output_path, embedding)
                    result = {"success": True, "path": output_path, "shape": embedding.shape}
                    print(f"  [OK] Saved to {output_filename}")
            except Exception as e:
                result = {"success": False, "error": str(e)}
                print(f"  [ERROR] {str(e)}")
            
            result["case_id"] = case_id
            result["image_id"] = image_id
            result["input_file"] = image_file
            results.append(result)
        
        return results
    
    def _print_summary(self, results):
        """Print summary of embedding generation."""
        print(f"\n{'='*60}")
        print("EMBEDDING ORCHESTRATION SUMMARY")
        print(f"{'='*60}")
        print(f"Staging Mode: {self.staging_mode}")
        print(f"Text Output Directory: {self.text_output_dir}")
        print(f"Image Output Directory: {self.image_output_dir}\n")
        
        # Text embeddings summary
        text_results = results["text"]
        if text_results:
            text_successful = sum(1 for r in text_results if r["success"])
            print(f"Text Embeddings: {text_successful}/{len(text_results)} successful")
            if text_successful < len(text_results):
                for r in text_results:
                    if not r["success"]:
                        print(f"  [FAIL] Case {r['case_id']:03d}: {r['error']}")
        else:
            print("Text Embeddings: No files processed")
        
        # Image embeddings summary
        image_results = results["images"]
        if image_results:
            image_successful = sum(1 for r in image_results if r["success"])
            print(f"Image Embeddings: {image_successful}/{len(image_results)} successful")
            if image_successful < len(image_results):
                for r in image_results:
                    if not r["success"]:
                        print(f"  [FAIL] Case {r['case_id']:03d}, Image {r['image_id']:03d}: {r['error']}")
        else:
            print("Image Embeddings: No files processed")
        
        # Overall summary
        total_successful = results["summary"]["text_successful"] + results["summary"]["image_successful"]
        total_files = len(text_results) + len(image_results)
        
        print(f"\nTotal: {total_successful}/{total_files} embeddings generated successfully")
        if total_successful == total_files:
            print("[OK] All embeddings generated and staged successfully!")
        
        print(f"{'='*60}\n")


def main():
    """Main entry point for the embedding orchestrator (production mode)."""
    orchestrator = EmbeddingOrchestrator(staging_mode="production")
    orchestrator.generate_all_embeddings()


if __name__ == "__main__":
    main()
