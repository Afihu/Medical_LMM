"""
Text Embedding Generator
Generates text embeddings from extracted text sections and saves them as .npy files.
"""

import os
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sentence_transformers import SentenceTransformer
from scripts.config.embedding_config import CONFIG, PATHS, NAMING_CONVENTIONS


class TextEmbeddingGenerator:
    """Generates embeddings from extracted text sections."""
    
    def __init__(self, model_name="NeuML/pubmedbert-base-embeddings", output_base_dir=None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
            output_base_dir: Base directory for outputs (default: project root)
        """
        self.model_name = model_name
        self.output_base = output_base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Verify embedding dimension
        self._verify_embedding_dimension()
        
        # Ensure output directory exists
        os.makedirs(os.path.join(self.output_base, PATHS["staged_embeddings"]["text"]), exist_ok=True)
    
    def _verify_embedding_dimension(self):
        """Verify that model produces expected embedding dimension."""
        test_embedding = self.model.encode("test")
        actual_dim = len(test_embedding)
        expected_dim = CONFIG["text_embedding_dim"]
        
        if actual_dim != expected_dim:
            print(f"[WARN] Embedding dimension mismatch!")
            print(f"  Expected: {expected_dim}, Got: {actual_dim}")
            if CONFIG["verify_on_first_run"]:
                raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}")
        else:
            print(f"[OK] Embedding dimension verified: {actual_dim}D")
    
    def _prepare_text_for_embedding(self, text_sections):
        """
        Prepare extracted text sections for embedding.
        
        Args:
            text_sections: Dictionary with keys like 'Title', 'History', etc.
            
        Returns:
            str: Combined text suitable for embedding
        """
        # Combine all sections with their labels
        parts = []
        for section, content in text_sections.items():
            if content:  # Only include non-empty sections
                parts.append(f"{section}: {content}")
        
        return " ".join(parts)
    
    def generate_embedding(self, text_sections):
        """
        Generate embedding from text sections.
        
        Args:
            text_sections: Dictionary with extracted text sections
            
        Returns:
            numpy.ndarray: Embedding vector of shape (embedding_dim,)
        """
        combined_text = self._prepare_text_for_embedding(text_sections)
        
        if not combined_text.strip():
            print("  [WARN] No text content to embed")
            return None
        
        embedding = self.model.encode(combined_text)
        return embedding
    
    def generate_and_save_embedding(self, text_json_path, case_id):
        """
        Generate embedding from extracted text JSON and save as .npy file.
        
        Args:
            text_json_path: Path to the extracted text JSON file
            case_id: Case ID for naming the output file
            
        Returns:
            dict: Result with status and path to saved embedding
        """
        try:
            # Load extracted text
            with open(text_json_path, 'r', encoding='utf-8') as f:
                text_sections = json.load(f)
            
            # Generate embedding
            embedding = self.generate_embedding(text_sections)
            
            if embedding is None:
                return {"success": False, "error": "No embedding generated"}
            
            # Save embedding
            output_filename = NAMING_CONVENTIONS["text_embedding"].format(case_id=case_id)
            output_path = os.path.join(self.output_base, PATHS["staged_embeddings"]["text"], output_filename)
            
            np.save(output_path, embedding)
            
            print(f"  [OK] Saved text embedding: {output_filename}")
            print(f"    Shape: {embedding.shape}, Path: {output_path}")
            
            return {"success": True, "path": output_path, "shape": embedding.shape}
        
        except Exception as e:
            error_msg = f"Error generating embedding from {text_json_path}: {str(e)}"
            print(f"  [ERROR] {error_msg}")
            return {"success": False, "error": error_msg}


def process_all_text_embeddings(input_dir=None, output_base_dir=None, model_name="NeuML/pubmedbert-base-embeddings"):
    """
    Process all extracted text files and generate embeddings.
    
    Args:
        input_dir: Directory containing extracted text JSON files
        output_base_dir: Base output directory for embeddings
        model_name: Sentence transformer model to use
        
    Returns:
        list: List of results for each embedding generated
    """
    project_root = output_base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    input_dir = input_dir or os.path.join(project_root, PATHS["extracted_data"]["text"])
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return []
    
    # Find all text JSON files
    text_files = [f for f in os.listdir(input_dir) if f.startswith("case_") and f.endswith("_text.json")]
    
    if not text_files:
        print(f"No text JSON files found in {input_dir}")
        return []
    
    print(f"\nFound {len(text_files)} text file(s) to embed")
    
    generator = TextEmbeddingGenerator(model_name, project_root)
    results = []
    
    print(f"\n{'='*60}")
    print("GENERATING TEXT EMBEDDINGS")
    print(f"{'='*60}\n")
    
    for text_file in sorted(text_files):
        # Extract case_id from filename (e.g., "case_001_text.json" -> 1)
        case_id = int(text_file.split("_")[1])
        
        text_path = os.path.join(input_dir, text_file)
        print(f"Processing: {text_file}")
        
        result = generator.generate_and_save_embedding(text_path, case_id)
        result["case_id"] = case_id
        result["input_file"] = text_file
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EMBEDDING GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    
    successful = sum(1 for r in results if r["success"])
    print(f"Successful: {successful}/{len(results)}")
    
    if successful < len(results):
        print("\nFailed embeddings:")
        for result in results:
            if not result["success"]:
                print(f"  [FAIL] Case {result['case_id']:03d}: {result['error']}")
    else:
        print("[OK] All embeddings generated successfully")
    
    return results


if __name__ == "__main__":
    # Example usage
    results = process_all_text_embeddings()
