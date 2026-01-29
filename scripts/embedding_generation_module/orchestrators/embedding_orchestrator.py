"""
Embedding Orchestrator (Embedding Pipeline)
Orchestrates the generation of both text and image embeddings from extracted data.
Handles model loading and manages where embeddings are staged.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from scripts.config.embedding_config import CONFIG, PATHS, NAMING_CONVENTIONS
from scripts.utils.model_manager import ModelManager


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
            self.caption_output_dir = os.path.join(self.output_base, PATHS["staged_embeddings"]["captions"])
        elif staging_mode == "temp":
            self.text_output_dir = os.path.join(self.output_base, "temp_query_data", "text_embeddings")
            self.image_output_dir = os.path.join(self.output_base, "temp_query_data", "image_embeddings")
            self.caption_output_dir = os.path.join(self.output_base, "temp_query_data", "caption_embeddings")
        else:
            raise ValueError(f"Unknown staging mode: {staging_mode}")
        
        # Ensure output directories exist
        os.makedirs(self.text_output_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(self.caption_output_dir, exist_ok=True)
        
        print(f"[INFO] Embedding Orchestrator initialized")
        print(f"  Staging Mode: {staging_mode}")
        print(f"  Text Output Directory: {self.text_output_dir}")
        print(f"  Image Output Directory: {self.image_output_dir}")
        print(f"  Caption Output Directory: {self.caption_output_dir}")
    
    def generate_all_embeddings(self):
        """
        Generate text, image, and caption embeddings from extracted data.
        Uses sequential processing: text → clear memory → images → clear memory → captions.
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
            "captions": [],
            "summary": {
                "text_successful": 0,
                "image_successful": 0,
                "caption_successful": 0,
                "total_successful": 0
            }
        }
        
        # PHASE 1: Generate text embeddings
        print("[PHASE 1/3] Generating text embeddings...")
        text_generator = ModelManager.get_text_generator()
        text_results = self._generate_text_embeddings(text_generator)
        results["text"] = text_results
        results["summary"]["text_successful"] = sum(1 for r in text_results if r["success"])
        
        # CLEANUP: Free text model from memory before loading image model
        print("\n[MEMORY CLEANUP] Releasing text embedding model from memory...")
        ModelManager.clear_models()
        print("[OK] Text model unloaded from RAM/VRAM\n")
        
        # PHASE 2: Generate image embeddings
        print("[PHASE 2/3] Generating image embeddings...")
        image_generator = ModelManager.get_image_generator()
        image_results = self._generate_image_embeddings(image_generator)
        results["images"] = image_results
        results["summary"]["image_successful"] = sum(1 for r in image_results if r["success"])
        
        # CLEANUP: Free image model from memory before loading caption model
        print("\n[MEMORY CLEANUP] Releasing image embedding model from memory...")
        ModelManager.clear_models()
        print("[OK] Image model unloaded from RAM/VRAM\n")
        
        # PHASE 3: Generate caption embeddings
        print("[PHASE 3/3] Generating caption embeddings...")
        caption_results = self._generate_caption_embeddings()
        results["captions"] = caption_results
        results["summary"]["caption_successful"] = sum(1 for r in caption_results if r["success"])
        
        # Final cleanup
        print("\n[MEMORY CLEANUP] Releasing models from memory...")
        ModelManager.clear_models()
        print("[OK] All models unloaded from RAM/VRAM\n")
        
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
    
    def _generate_caption_embeddings(self):
        """
        Generate caption embeddings using the MedSigLip dual encoder.
        Since MedSigLip is a dual encoder (image+text), we use it to encode captions as text.
        """
        import json
        import numpy as np
        from scripts.embedding_generation_module.generators.image_embedding_generator import ImageEmbeddingGenerator
        
        caption_dir = os.path.join(self.output_base, PATHS["extracted_data"]["captions"])
        
        if not os.path.exists(caption_dir):
            print(f"  [WARN] Caption input directory not found: {caption_dir}")
            return []
        
        caption_files = sorted([f for f in os.listdir(caption_dir) if f.startswith("case_") and f.endswith("_captions.json")])
        
        if not caption_files:
            print(f"  [WARN] No caption files found in {caption_dir}")
            return []
        
        # Initialize image embedding generator (MedSigLip dual encoder handles both images and text)
        image_generator = ImageEmbeddingGenerator(output_base_dir=self.output_base)
        results = []
        
        for caption_file in caption_files:
            # Extract case_id from filename
            case_id = int(caption_file.split("_")[1])
            caption_path = os.path.join(caption_dir, caption_file)
            
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_data = json.load(f)
                
                if not caption_data.get("captions"):
                    print(f"  [WARN] No captions found in {caption_file}")
                    continue
                
                print(f"  Processing: {caption_file}")
                
                # Generate embeddings for each caption using the text encoder of MedSigLip
                for caption_entry in caption_data["captions"]:
                    caption_id = caption_entry.get("id")
                    caption_text = caption_entry.get("text", "")
                    
                    if not caption_text.strip():
                        result = {"success": False, "error": "Empty caption text", "case_id": case_id, "caption_id": caption_id}
                    else:
                        try:
                            # Use the processor to encode text only
                            inputs = image_generator.processor(text=[caption_text], return_tensors="pt", padding="max_length").to(image_generator.device)
                            
                            import torch
                            with torch.no_grad():
                                outputs = image_generator.model.get_text_features(**inputs)
                            
                            embedding = outputs[0].cpu().numpy()
                            
                            # Save embedding
                            output_filename = NAMING_CONVENTIONS["caption_embedding"].format(case_id=case_id, caption_id=caption_id)
                            output_path = os.path.join(self.caption_output_dir, output_filename)
                            
                            np.save(output_path, embedding)
                            result = {"success": True, "path": output_path, "shape": embedding.shape}
                            print(f"    [OK] Saved caption {caption_id}: {output_filename}")
                        except Exception as e:
                            result = {"success": False, "error": str(e)}
                            print(f"    [ERROR] {str(e)}")
                    
                    result["case_id"] = case_id
                    result["caption_id"] = caption_id
                    results.append(result)
            
            except Exception as e:
                print(f"  [ERROR] Failed to process {caption_file}: {e}")
                results.append({"success": False, "error": str(e), "case_id": case_id})
        
        return results
    
    def _print_summary(self, results):
        """Print summary of embedding generation."""
        print(f"\n{'='*60}")
        print("EMBEDDING ORCHESTRATION SUMMARY")
        print(f"{'='*60}")
        print(f"Staging Mode: {self.staging_mode}")
        print(f"Text Output Directory: {self.text_output_dir}")
        print(f"Image Output Directory: {self.image_output_dir}")
        print(f"Caption Output Directory: {self.caption_output_dir}\n")
        
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
        
        # Caption embeddings summary
        caption_results = results.get("captions", [])
        if caption_results:
            caption_successful = sum(1 for r in caption_results if r["success"])
            print(f"Caption Embeddings: {caption_successful}/{len(caption_results)} successful")
            if caption_successful < len(caption_results):
                for r in caption_results:
                    if not r["success"]:
                        print(f"  [FAIL] Case {r['case_id']:03d}, Caption {r.get('caption_id', '?')}: {r['error']}")
        else:
            print("Caption Embeddings: No files processed")
        
        # Overall summary
        total_successful = (results["summary"]["text_successful"] + 
                           results["summary"]["image_successful"] + 
                           results["summary"].get("caption_successful", 0))
        total_files = len(text_results) + len(image_results) + len(caption_results)
        
        print(f"\nTotal: {total_successful}/{total_files} embeddings generated successfully")
        if total_files > 0 and total_successful == total_files:
            print("[OK] All embeddings generated and staged successfully!")
        
        print(f"{'='*60}\n")


def main():
    """Main entry point for the embedding orchestrator (production mode)."""
    orchestrator = EmbeddingOrchestrator(staging_mode="production")
    orchestrator.generate_all_embeddings()


if __name__ == "__main__":
    main()
