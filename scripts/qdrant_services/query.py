"""
query.py (Refactored)
------------------------------------------------
Performs similarity search on Qdrant collections for text and/or image embeddings.
Saves retrieved cases under /temp_query_data/<session_id>/retrieved_cases/.

Implements logic from MAJOR_REFRACTOR.md §Main Runtime Pipeline → Qdrant Query Module.
"""

import os
import json
import shutil
from datetime import datetime
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# --- Path configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TEMP_QUERY_DATA = os.path.join(PROJECT_ROOT, "temp_query_data")
os.makedirs(TEMP_QUERY_DATA, exist_ok=True)

# --- Load credentials ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME_TEXT = "medical_case_texts"  # Change to your Qdrant collection name
COLLECTION_NAME_IMAGE = "medical_case_images"

# -------------------------------------------------------------------

def setup_qdrant_client():
    """Connect to Qdrant Cloud instance."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"[OK] Connected to Qdrant Cloud at {QDRANT_URL}")
    return client

def query_collection(client, vector, collection_name, top_k=5):
    """Perform similarity search and return top results."""
    print(f"[INFO] Querying collection '{collection_name}' ...")
    results = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k,
        with_payload=True
    )
    print(f"[OK] Retrieved {len(results)} results from '{collection_name}'.")
    return results

def merge_results(text_results, image_results):
    """
    Merge text and image search results by common case_id.
    Returns a dict grouped by case_id, consistent with MAJOR_REFRACTOR.md.
    """
    print("[INFO] Merging Qdrant query results...")

    retrieved_cases = {}

    # --- Process text results ---
    for res in text_results:
        cid = res.payload.get("case_id")
        if cid not in retrieved_cases:
            retrieved_cases[cid] = {
                "case_id": cid,
                "text": res.payload,
                "images": []
            }

    # --- Process image results ---
    for res in image_results:
        cid = res.payload.get("case_id")
        if cid in retrieved_cases:
            retrieved_cases[cid]["images"].append(res.payload)
        else:
            # case_id appears only in image results
            retrieved_cases[cid] = {
                "case_id": cid,
                "text": {},
                "images": [res.payload]
            }

    print(f"[OK] Merged {len(retrieved_cases)} unique cases.")
    return retrieved_cases

def save_retrieved_cases(retrieved_cases, session_id="default_session"):
    """Save retrieved cases as JSON files under temp_query_data/<session_id>/retrieved_cases/."""
    session_dir = os.path.join(TEMP_QUERY_DATA, session_id, "retrieved_cases")
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    os.makedirs(session_dir, exist_ok=True)

    for idx, (cid, data) in enumerate(retrieved_cases.items(), start=1):
        out_path = os.path.join(session_dir, f"case_{idx:03d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[OK] Saved {len(retrieved_cases)} merged cases to '{session_dir}'")
    return session_dir

def run_query(text_vector=None, image_vector=None, top_k=5, session_id=None):
    """
    Orchestrate Qdrant query for one or both modalities.
    Args:
        text_vector: numpy array or list for text query (optional)
        image_vector: numpy array or list for image query (optional)
        top_k: number of results to return
        session_id: session identifier (for temp data path)
    
    Returns:
        tuple: (retrieved_cases dict, saved_dir path) or (None, None) if no vectors provided
    """
    # Validate that at least one vector is provided
    if text_vector is None and image_vector is None:
        print("[WARN] No query vectors provided — nothing to search.")
        return None, None
    
    client = setup_qdrant_client()
    text_results = []
    image_results = []

    # Query text modality if provided
    if text_vector is not None:
        print("[INFO] Text vector provided. Querying text collection...")
        text_results = query_collection(client, text_vector, COLLECTION_NAME_TEXT, top_k)
    else:
        print("[INFO] No text vector provided. Skipping text search.")
    
    # Query image modality if provided
    if image_vector is not None:
        print("[INFO] Image vector provided. Querying image collection...")
        image_results = query_collection(client, image_vector, COLLECTION_NAME_IMAGE, top_k)
    else:
        print("[INFO] No image vector provided. Skipping image search.")

    # Merge results and return
    retrieved_cases = merge_results(text_results, image_results)
    
    if not retrieved_cases:
        print("[WARN] No cases retrieved from any modality.")
        return None, None
    
    saved_dir = save_retrieved_cases(retrieved_cases, session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[OK] Query complete. Retrieved {len(retrieved_cases)} cases total.")
    return retrieved_cases, saved_dir



if __name__ == "__main__":
    # Example run (for testing)
    import numpy as np
    dummy_vec = np.random.rand(768).tolist()  # Example vector
    run_query(text_vector=dummy_vec, top_k=3, session_id="test_session")
