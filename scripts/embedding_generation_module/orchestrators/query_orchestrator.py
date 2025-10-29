"""
Query Orchestrator (Main Pipeline)
Orchestrates the generation of embeddings for user queries.
Reuses embedding generators from the Embedding Pipeline.
Stages embeddings to temp_query_data/ for temporary use during a session.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from PIL import Image

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from scripts.config.embedding_config import CONFIG, PATHS, NAMING_CONVENTIONS
from scripts.embedding_generation_module.orchestrators.embedding_orchestrator import ModelManager


class QueryOrchestrator:
    """
    Orchestrates embedding generation for user queries in the Main Pipeline.
    Uses ModelManager to reuse models from the Embedding Pipeline.
    Stages embeddings temporarily for the current session.
    """
    
    def __init__(self, output_base_dir=None, session_id=None):
        """
        Initialize the query orchestrator.
        
        Args:
            output_base_dir: Base directory for outputs (default: project root)
            session_id: Unique session identifier for temp data organization
        """
        self.output_base = output_base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        self.session_id = session_id or "default_session"
        
        # Temp directory for this session
        self.temp_dir = os.path.join(self.output_base, "temp_query_data", self.session_id)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        print(f"[INFO] Query Orchestrator initialized")
        print(f"  Session ID: {self.session_id}")
        print(f"  Temp Directory: {self.temp_dir}")
    
    def embed_text_query(self, query_text):
        """
        Embed a text query using the text embedding generator.
        
        Args:
            query_text: User's text query
            
        Returns:
            numpy.ndarray: Text embedding vector
        """
        print(f"[INFO] Embedding text query...")
        text_generator = ModelManager.get_text_generator(output_base_dir=self.output_base)
        
        # Prepare as dictionary format (matching extracted_data format)
        query_dict = {
            "Title": query_text,
            "History": "",
            "Clinical Findings": "",
            "Discussion": "",
            "Summary Box First Line": ""
        }
        
        embedding = text_generator.generate_embedding(query_dict)
        
        if embedding is not None:
            print(f"[OK] Text query embedded (shape: {embedding.shape})")
            # Save to temp for reference
            temp_embedding_path = os.path.join(self.temp_dir, "query_text_embedding.npy")
            import numpy as np
            np.save(temp_embedding_path, embedding)
        
        return embedding
    
    def embed_image_query(self, image_path, caption_text=None):
        """
        Embed an image query using the image embedding generator.
        
        Args:
            image_path: Path to the query image
            caption_text: Optional caption for the image
            
        Returns:
            numpy.ndarray: Image embedding vector
        """
        print(f"[INFO] Embedding image query...")
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file not found: {image_path}")
            return None
        
        if not caption_text:
            caption_text = "Query image"
        
        image_generator = ModelManager.get_image_generator(output_base_dir=self.output_base)
        embedding = image_generator.generate_embedding(image_path, caption_text)
        
        if embedding is not None:
            print(f"[OK] Image query embedded (shape: {embedding.shape})")
            # Save to temp for reference
            temp_embedding_path = os.path.join(self.temp_dir, "query_image_embedding.npy")
            import numpy as np
            np.save(temp_embedding_path, embedding)
        
        return embedding
    
    def cleanup_session(self):
        """Clean up temporary data for this session."""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"[OK] Cleaned up session data: {self.temp_dir}")
            except Exception as e:
                print(f"[WARN] Could not clean up session data: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto cleanup on exit."""
        if CONFIG.get("temp_data_cleanup", True):
            self.cleanup_session()


# Example usage with context manager
if __name__ == "__main__":
    with QueryOrchestrator(session_id="test_session") as orchestrator:
        # Embed a text query
        text_embedding = orchestrator.embed_text_query("What are the symptoms of Ebola?")
        
        # Embed an image query (example)
        # image_embedding = orchestrator.embed_image_query("path/to/image.png", "Medical image")
        
        print(f"\nSession data saved to: {orchestrator.temp_dir}")
