"""
Unified Qdrant Upload Service
==============================
Uploads both text and image embeddings to Qdrant Cloud using the v2 API.

This service:
1. Loads staged embeddings from ./staged_embeddings/
2. Retrieves corresponding case metadata (text content)
3. Creates Qdrant points with standardized payload structure
4. Uploads to two separate collections:
   - medical_case_texts: Text embeddings with full case metadata
   - medical_case_images: Image embeddings with case metadata and captions

Point Structure (as per MAJOR_REFRACTOR.md):
- Text: {id, vector, payload: {case_id, Title, History, Clinical_Findings, Discussion, Summary_Box_First_Line}}
- Image: {id, vector, payload: {case_id, image_id, Caption, Title, History, Clinical_Findings, Discussion, Summary_Box_First_Line}}
"""

import os
import json
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.config.embedding_config import CONFIG, PATHS, NAMING_CONVENTIONS


class QdrantUploadService:
    """Service for uploading embeddings to Qdrant Cloud."""
    
    # Collection names (standardized)
    COLLECTION_TEXT = "medical_case_texts"
    COLLECTION_IMAGE = "medical_case_images"
    
    def __init__(self, qdrant_url=None, qdrant_api_key=None):
        """
        Initialize the Qdrant upload service.
        
        Args:
            qdrant_url: Qdrant Cloud URL (default: from QDRANT_URL_v2 env var)
            qdrant_api_key: Qdrant API key (default: from QDRANT_API_KEY_v2 env var)
        """
        # Load environment variables
        load_dotenv()
        
        # Use v2 credentials by default (newer API)
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL_v2")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY_v2")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError(
                "Missing Qdrant credentials. Ensure QDRANT_URL_v2 and QDRANT_API_KEY_v2 "
                "are set in .env file"
            )
        
        self.client = self._setup_client()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    def _setup_client(self) -> QdrantClient:
        """Connect to Qdrant Cloud."""
        try:
            client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
            print(f"[OK] Connected to Qdrant Cloud")
            print(f"  URL: {self.qdrant_url}")
            return client
        except Exception as e:
            print(f"[ERROR] Failed to connect to Qdrant: {e}")
            raise
    
    def _ensure_collection(self, collection_name: str, vector_size: int, named_vectors: bool = False) -> None:
        """
        Create collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors
            named_vectors: If True, create collection with named vectors for image embeddings
                          (both "image" and "caption" vectors)
        """
        try:
            collections = [col.name for col in self.client.get_collections().collections]
            if collection_name in collections:
                print(f"[INFO] Collection '{collection_name}' already exists")
                return
            
            if named_vectors and collection_name == self.COLLECTION_IMAGE:
                # For image embeddings, use named vectors to support both image and caption embeddings
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "image": models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        ),
                        "caption": models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        ),
                    },
                )
                print(f"[OK] Created collection '{collection_name}' with named vectors (image & caption, {vector_size}D)")
            else:
                # For text embeddings, use single vector
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                print(f"[OK] Created collection '{collection_name}' (vector_size={vector_size}D)")
        except Exception as e:
            print(f"[ERROR] Failed to create/check collection '{collection_name}': {e}")
            raise
    
    def _load_extracted_text(self, case_id: int) -> Dict[str, str]:
        """
        Load extracted text sections for a case.
        
        Args:
            case_id: Case ID
            
        Returns:
            Dictionary with text sections (Title, History, Clinical_Findings, Discussion, Summary_Box_First_Line)
        """
        text_filename = NAMING_CONVENTIONS["text"].format(case_id=case_id)
        text_path = os.path.join(self.project_root, PATHS["extracted_data"]["text"], text_filename)
        
        default_payload = {
            "Title": "",
            "History": "",
            "Clinical_Findings": "",
            "Discussion": "",
            "Summary_Box_First_Line": "",
        }
        
        if not os.path.exists(text_path):
            print(f"    [WARN] Text file not found: {text_filename}")
            return default_payload
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
            
            # Convert spaces to underscores in keys to match Qdrant payload format
            payload = {}
            for key, value in text_data.items():
                key_normalized = key.replace(" ", "_")
                payload[key_normalized] = str(value) if value else ""
            
            return payload
        except Exception as e:
            print(f"    [WARN] Error loading text file {text_filename}: {e}")
            return default_payload
    
    def _load_image_caption(self, case_id: int, image_id: int) -> str:
        """
        Load caption for a specific image.
        
        Args:
            case_id: Case ID
            image_id: Image ID
            
        Returns:
            Caption text or empty string
        """
        caption_filename = f"case_{case_id:03d}_captions.json"
        caption_path = os.path.join(self.project_root, PATHS["extracted_data"]["captions"], caption_filename)
        
        if not os.path.exists(caption_path):
            return ""
        
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            
            # Find caption matching image_id
            if caption_data.get("captions"):
                for cap in caption_data["captions"]:
                    if cap.get("id") == image_id:
                        return cap.get("text", "")
            
            return ""
        except Exception as e:
            print(f"    [WARN] Error loading caption file {caption_filename}: {e}")
            return ""
    
    def _load_text_embeddings(self) -> List[Tuple[str, np.ndarray, int]]:
        """
        Load all text embeddings from staged_embeddings/text_embeddings/.
        
        Returns:
            List of (filename, embedding_vector, case_id)
        """
        text_emb_dir = os.path.join(self.project_root, PATHS["staged_embeddings"]["text"])
        
        if not os.path.exists(text_emb_dir):
            print(f"[WARN] Text embeddings directory not found: {text_emb_dir}")
            return []
        
        embeddings = []
        npy_files = sorted([f for f in os.listdir(text_emb_dir) if f.endswith("_text_embedding.npy")])
        
        print(f"[INFO] Found {len(npy_files)} text embedding file(s)")
        
        for filename in npy_files:
            try:
                # Extract case_id from filename: case_001_text_embedding.npy
                case_id = int(filename.split("_")[1])
                filepath = os.path.join(text_emb_dir, filename)
                embedding = np.load(filepath)
                embeddings.append((filename, embedding, case_id))
                print(f"  [OK] Loaded {filename} (case_id={case_id:03d}, shape={embedding.shape})")
            except Exception as e:
                print(f"  [ERROR] Failed to load {filename}: {e}")
        
        return embeddings
    
    def _load_image_embeddings(self) -> List[Tuple[str, np.ndarray, int, int]]:
        """
        Load all image embeddings from staged_embeddings/image_embeddings/.
        
        Returns:
            List of (filename, embedding_vector, case_id, image_id)
        """
        image_emb_dir = os.path.join(self.project_root, PATHS["staged_embeddings"]["images"])
        
        if not os.path.exists(image_emb_dir):
            print(f"[WARN] Image embeddings directory not found: {image_emb_dir}")
            return []
        
        embeddings = []
        npy_files = sorted([f for f in os.listdir(image_emb_dir) if f.endswith("_embedding.npy")])
        
        print(f"[INFO] Found {len(npy_files)} image embedding file(s)")
        
        for filename in npy_files:
            try:
                # Extract case_id and image_id from filename: case_001_image_001_embedding.npy
                parts = filename.replace("_embedding.npy", "").split("_")
                case_id = int(parts[1])
                image_id = int(parts[3])
                filepath = os.path.join(image_emb_dir, filename)
                embedding = np.load(filepath)
                embeddings.append((filename, embedding, case_id, image_id))
                print(f"  [OK] Loaded {filename} (case_id={case_id:03d}, image_id={image_id:03d}, shape={embedding.shape})")
            except Exception as e:
                print(f"  [ERROR] Failed to load {filename}: {e}")
        
        return embeddings
    
    def _load_caption_embeddings(self) -> List[Tuple[str, np.ndarray, int, int]]:
        """
        Load all caption embeddings from staged_embeddings/caption_embeddings/.
        
        Returns:
            List of (filename, embedding_vector, case_id, caption_id)
        """
        caption_emb_dir = os.path.join(self.project_root, PATHS["staged_embeddings"]["captions"])
        
        if not os.path.exists(caption_emb_dir):
            print(f"[INFO] Caption embeddings directory not found: {caption_emb_dir} (optional)")
            return []
        
        embeddings = []
        npy_files = sorted([f for f in os.listdir(caption_emb_dir) if f.endswith("_embedding.npy")])
        
        print(f"[INFO] Found {len(npy_files)} caption embedding file(s)")
        
        for filename in npy_files:
            try:
                # Extract case_id and caption_id from filename: case_001_caption_001_embedding.npy
                parts = filename.replace("_embedding.npy", "").split("_")
                case_id = int(parts[1])
                caption_id = int(parts[3])
                filepath = os.path.join(caption_emb_dir, filename)
                embedding = np.load(filepath)
                embeddings.append((filename, embedding, case_id, caption_id))
                print(f"  [OK] Loaded {filename} (case_id={case_id:03d}, caption_id={caption_id:03d}, shape={embedding.shape})")
            except Exception as e:
                print(f"  [ERROR] Failed to load {filename}: {e}")
        
        return embeddings
    
    def upload_text_embeddings(self) -> Dict:
        """
        Upload text embeddings to Qdrant.
        
        Returns:
            Dictionary with upload statistics
        """
        print(f"\n{'='*70}")
        print("UPLOADING TEXT EMBEDDINGS")
        print(f"{'='*70}\n")
        
        # Load embeddings
        embeddings = self._load_text_embeddings()
        if not embeddings:
            print("[WARN] No text embeddings found to upload")
            return {"success": False, "uploaded": 0, "total": 0}
        
        # Ensure collection
        vector_size = len(embeddings[0][1])  # Size of first embedding
        self._ensure_collection(self.COLLECTION_TEXT, vector_size)
        
        # Create points
        points = []
        for filename, embedding, case_id in embeddings:
            try:
                # Load text sections for payload
                text_payload = self._load_extracted_text(case_id)
                
                # Create point with standardized structure
                point = models.PointStruct(
                    id=str(uuid.uuid4()),  # Qdrant generates UUIDs
                    vector=embedding.tolist(),
                    payload={
                        "case_id": case_id,
                        **text_payload  # Merge text sections into payload
                    }
                )
                points.append(point)
                print(f"  [OK] Created point for case_{case_id:03d}")
            except Exception as e:
                print(f"  [ERROR] Failed to create point from {filename}: {e}")
        
        # Upload points
        if points:
            try:
                self.client.upsert(
                    collection_name=self.COLLECTION_TEXT,
                    points=points
                )
                print(f"\n[OK] Uploaded {len(points)} text embeddings to '{self.COLLECTION_TEXT}'")
                return {"success": True, "uploaded": len(points), "total": len(embeddings)}
            except Exception as e:
                print(f"[ERROR] Failed to upload points: {e}")
                return {"success": False, "uploaded": 0, "total": len(embeddings)}
        else:
            print("[WARN] No valid points to upload")
            return {"success": False, "uploaded": 0, "total": len(embeddings)}
    
    def upload_image_embeddings(self) -> Dict:
        """
        Upload image embeddings to Qdrant with associated caption embeddings.
        Uses named vectors to store both image and caption embeddings in the same point.
        
        Returns:
            Dictionary with upload statistics
        """
        print(f"\n{'='*70}")
        print("UPLOADING IMAGE EMBEDDINGS (with captions)")
        print(f"{'='*70}\n")
        
        # Load embeddings
        image_embeddings = self._load_image_embeddings()
        if not image_embeddings:
            print("[WARN] No image embeddings found to upload")
            return {"success": False, "uploaded": 0, "total": 0}
        
        # Load caption embeddings
        caption_embeddings = self._load_caption_embeddings()
        
        # Create a map of caption_id to embedding for quick lookup: (case_id, caption_id) -> embedding
        caption_map = {}
        for filename, embedding, case_id, caption_id in caption_embeddings:
            caption_map[(case_id, caption_id)] = embedding
        
        # Ensure collection with named vectors
        vector_size = len(image_embeddings[0][1])  # Size of image embedding
        self._ensure_collection(self.COLLECTION_IMAGE, vector_size, named_vectors=True)
        
        # Create points
        points = []
        for filename, image_embedding, case_id, image_id in image_embeddings:
            try:
                # Load text sections for payload (shared across images of same case)
                text_payload = self._load_extracted_text(case_id)
                
                # Load caption text for this specific image
                caption_text = self._load_image_caption(case_id, image_id)
                
                # Try to get caption embedding for this image
                caption_embedding = caption_map.get((case_id, image_id))
                
                # Build payload
                payload = {
                    "case_id": case_id,
                    "image_id": image_id,
                    "Caption": caption_text,
                    **text_payload  # Merge text sections into payload
                }
                
                # Create named vectors dict with image and caption vectors
                vectors = {
                    "image": image_embedding.tolist(),
                }
                
                if caption_embedding is not None:
                    vectors["caption"] = caption_embedding.tolist()
                    payload["has_caption_embedding"] = True
                    print(f"  [OK] case_{case_id:03d}, image_{image_id:03d}: with caption embedding")
                else:
                    # If no caption embedding, use image embedding for caption vector as fallback
                    vectors["caption"] = image_embedding.tolist()
                    payload["has_caption_embedding"] = False
                    print(f"  [WARN] case_{case_id:03d}, image_{image_id:03d}: no caption embedding (using image embedding)")
                
                # Create point with named vectors structure
                point = models.PointStruct(
                    id=str(uuid.uuid4()),  # Qdrant generates UUIDs
                    vector=vectors,
                    payload=payload
                )
                points.append(point)
            except Exception as e:
                print(f"  [ERROR] Failed to create point from {filename}: {e}")
        
        # Upload points
        if points:
            try:
                self.client.upsert(
                    collection_name=self.COLLECTION_IMAGE,
                    points=points
                )
                print(f"\n[OK] Uploaded {len(points)} image embeddings to '{self.COLLECTION_IMAGE}'")
                return {"success": True, "uploaded": len(points), "total": len(image_embeddings)}
            except Exception as e:
                print(f"[ERROR] Failed to upload points: {e}")
                return {"success": False, "uploaded": 0, "total": len(image_embeddings)}
        else:
            print("[WARN] No valid points to upload")
            return {"success": False, "uploaded": 0, "total": len(image_embeddings)}
    
    def upload_all_embeddings(self) -> Dict:
        """
        Upload both text and image embeddings to Qdrant.
        
        Returns:
            Dictionary with combined upload statistics
        """
        print(f"\n{'='*70}")
        print("QDRANT UPLOAD SERVICE - STARTING")
        print(f"{'='*70}")
        
        # Upload text embeddings
        text_results = self.upload_text_embeddings()
        
        # Upload image embeddings
        image_results = self.upload_image_embeddings()
        
        # Print summary
        print(f"\n{'='*70}")
        print("UPLOAD SUMMARY")
        print(f"{'='*70}")
        print(f"Text Embeddings: {text_results['uploaded']}/{text_results['total']} uploaded")
        print(f"Image Embeddings: {image_results['uploaded']}/{image_results['total']} uploaded")
        
        total_uploaded = text_results["uploaded"] + image_results["uploaded"]
        total_files = text_results["total"] + image_results["total"]
        
        print(f"\nTotal: {total_uploaded}/{total_files} embeddings uploaded successfully")
        
        if total_uploaded == total_files and total_files > 0:
            print("[OK] All embeddings uploaded successfully!")
        elif total_uploaded > 0:
            print(f"[WARN] Partial upload: {total_files - total_uploaded} file(s) failed")
        else:
            print("[ERROR] No embeddings were uploaded")
        
        print(f"{'='*70}\n")
        
        return {
            "text": text_results,
            "image": image_results,
            "total_uploaded": total_uploaded,
            "total_files": total_files,
            "success": total_uploaded == total_files and total_files > 0
        }


def main():
    """Main entry point for the upload service."""
    try:
        service = QdrantUploadService()
        results = service.upload_all_embeddings()
        
        # Exit with appropriate code
        exit(0 if results["success"] else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        exit(1)


if __name__ == "__main__":
    main()
