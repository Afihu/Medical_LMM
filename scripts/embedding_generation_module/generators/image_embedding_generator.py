"""
Image Embedding Generator
Generates image embeddings from extracted images and captions using MedSigLip model.
"""

import os
import json
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from scripts.config.embedding_config import CONFIG, PATHS, NAMING_CONVENTIONS


class ImageEmbeddingGenerator:
    """Generates embeddings from images and captions using MedSigLip model."""
    
    def __init__(self, model_name="google/medsiglip-448", output_base_dir=None):
        """
        Initialize the image embedding generator.
        
        Args:
            model_name: Name of the model to use for image embedding
            output_base_dir: Base directory for outputs (default: project root)
        """
        self.model_name = model_name
        self.output_base = output_base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading image embedding model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load with HuggingFace token for gated models
        hf_token = CONFIG.get("huggingface_token")
        self.model = AutoModel.from_pretrained(model_name, token=hf_token).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        
        # Verify embedding dimension
        self._verify_embedding_dimension()
        
        # Ensure output directory exists
        os.makedirs(os.path.join(self.output_base, PATHS["staged_embeddings"]["images"]), exist_ok=True)
    
    def _verify_embedding_dimension(self):
        """Verify that model produces expected embedding dimension."""
        # Create dummy inputs to test
        with torch.no_grad():
            dummy_image = Image.new('RGB', (448, 448))
            dummy_text = "test"
            inputs = self.processor(text=[dummy_text], images=[dummy_image], padding="max_length", return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            actual_dim = outputs.image_embeds.shape[-1]
        
        expected_dim = CONFIG["image_embedding_dim"]
        
        if actual_dim != expected_dim:
            print(f"[WARN] Embedding dimension mismatch!")
            print(f"  Expected: {expected_dim}, Got: {actual_dim}")
            if CONFIG["verify_on_first_run"]:
                raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}")
        else:
            print(f"[OK] Image embedding dimension verified: {actual_dim}D")
    
    def generate_embedding(self, image_path, caption_text=None):
        """
        Generate embedding from an image and optional caption.
        
        Args:
            image_path: Path to the image file (.png)
            caption_text: Caption text for the image (optional, but recommended for better embeddings)
            
        Returns:
            numpy.ndarray: Image embedding vector of shape (embedding_dim,)
            
        Note:
            - If caption_text is None or empty, only image will be used
            - For best results, provide meaningful captions
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Use caption if provided and non-empty, otherwise use image-only encoding
            if not caption_text or caption_text.strip() == "":
                # Image-only encoding: use empty string for text
                caption_text = ""
                print(f"    [WARN] No caption provided, using image-only encoding")
            
            # Generate embedding
            inputs = self.processor(text=[caption_text], images=[image], padding="max_length", return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract and convert to numpy
            image_embedding = outputs.image_embeds[0].cpu().numpy()
            return image_embedding
        
        except Exception as e:
            print(f"  [ERROR] Error generating embedding for {image_path}: {e}")
            return None
    
    def generate_and_save_embedding(self, image_path, caption_text=None, case_id=None, image_id=None):
        """
        Generate embedding from image and optional caption, then save as .npy file.
        
        Args:
            image_path: Path to the extracted image PNG file
            caption_text: Caption text for the image (optional, but recommended)
            case_id: Case ID for naming
            image_id: Image ID for naming
            
        Returns:
            dict: Result with status and path to saved embedding
        """
        try:
            # Generate embedding (works with or without caption)
            embedding = self.generate_embedding(image_path, caption_text)
            
            if embedding is None:
                return {"success": False, "error": "No embedding generated"}
            
            # Save embedding
            output_filename = NAMING_CONVENTIONS["image_embedding"].format(case_id=case_id, image_id=image_id)
            output_path = os.path.join(self.output_base, PATHS["staged_embeddings"]["images"], output_filename)
            
            np.save(output_path, embedding)
            
            caption_status = "with caption" if caption_text and caption_text.strip() else "image-only"
            print(f"  [OK] Saved image embedding ({caption_status}): {output_filename}")
            print(f"    Shape: {embedding.shape}, Path: {output_path}")
            
            return {"success": True, "path": output_path, "shape": embedding.shape}
        
        except Exception as e:
            error_msg = f"Error saving image embedding: {str(e)}"
            print(f"  [ERROR] {error_msg}")
            return {"success": False, "error": error_msg}


def process_all_image_embeddings(input_dir_images=None, input_dir_captions=None, output_base_dir=None, model_name="google/medsiglip-448"):
    """
    Process all extracted image files and their captions to generate embeddings.
    
    Args:
        input_dir_images: Directory containing extracted image PNG files
        input_dir_captions: Directory containing extracted caption JSON files
        output_base_dir: Base output directory for embeddings
        model_name: Model to use for image embedding
        
    Returns:
        list: List of results for each embedding generated
    """
    project_root = output_base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    input_dir_images = input_dir_images or os.path.join(project_root, PATHS["extracted_data"]["images"])
    input_dir_captions = input_dir_captions or os.path.join(project_root, PATHS["extracted_data"]["captions"])
    
    if not os.path.exists(input_dir_images):
        print(f"Error: Image directory '{input_dir_images}' does not exist")
        return []
    
    if not os.path.exists(input_dir_captions):
        print(f"Error: Caption directory '{input_dir_captions}' does not exist")
        return []
    
    # Find all image PNG files
    image_files = sorted([f for f in os.listdir(input_dir_images) if f.startswith("case_") and f.endswith("_image_") and f.endswith(".png")])
    
    if not image_files:
        print(f"No image PNG files found in {input_dir_images}")
        return []
    
    print(f"\nFound {len(image_files)} image file(s) to embed")
    
    generator = ImageEmbeddingGenerator(model_name, project_root)
    results = []
    
    print(f"\n{'='*60}")
    print("GENERATING IMAGE EMBEDDINGS")
    print(f"{'='*60}\n")
    
    for image_file in image_files:
        # Extract case_id and image_id from filename (e.g., "case_001_image_001.png")
        parts = image_file.replace(".png", "").split("_")
        case_id = int(parts[1])
        image_id = int(parts[3])
        
        image_path = os.path.join(input_dir_images, image_file)
        
        # Load caption for this image (optional but recommended)
        caption_text = None
        caption_file = f"case_{case_id:03d}_captions.json"
        caption_path = os.path.join(input_dir_captions, caption_file)
        
        if os.path.exists(caption_path):
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_data = json.load(f)
                    if caption_data.get("captions"):
                        # Find caption matching image_id
                        for cap in caption_data["captions"]:
                            if cap.get("id") == image_id:
                                caption_text = cap.get("text", "").strip()
                                break
            except Exception as e:
                print(f"  [WARN] Could not load caption from {caption_file}: {e}")
        
        # Caption is optional - proceed even if not found
        if caption_text:
            print(f"  Caption found: '{caption_text[:50]}...'")
        else:
            print(f"  [WARN] No caption available - will use image-only encoding")
        
        print(f"Processing: {image_file}")
        result = generator.generate_and_save_embedding(image_path, caption_text, case_id, image_id)
        result["case_id"] = case_id
        result["image_id"] = image_id
        result["input_file"] = image_file
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("IMAGE EMBEDDING GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    
    successful = sum(1 for r in results if r["success"])
    print(f"Successful: {successful}/{len(results)}")
    
    if successful < len(results):
        print("\nFailed embeddings:")
        for result in results:
            if not result["success"]:
                print(f"  [FAIL] Case {result['case_id']:03d}, Image {result['image_id']:03d}: {result['error']}")
    else:
        print("[OK] All image embeddings generated successfully")
    
    return results


if __name__ == "__main__":
    # Example usage
    results = process_all_image_embeddings()
