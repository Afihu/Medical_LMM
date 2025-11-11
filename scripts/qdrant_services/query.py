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

# Import query configuration
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.config.query_config import QUERY_CONFIG, QDRANT_COLLECTIONS

# --- Path configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TEMP_QUERY_DATA = os.path.join(PROJECT_ROOT, "temp_query_data")
os.makedirs(TEMP_QUERY_DATA, exist_ok=True)

# --- Load credentials ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME_TEXT = QDRANT_COLLECTIONS.get("text", "medical_case_texts")
COLLECTION_NAME_IMAGE = QDRANT_COLLECTIONS.get("image", "medical_case_images")

# -------------------------------------------------------------------

def setup_qdrant_client():
    """Connect to Qdrant Cloud instance."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"[OK] Connected to Qdrant Cloud at {QDRANT_URL}")
    return client

def query_collection(client, vector, collection_name, top_k=5, vector_name=None):
    """
    Perform similarity search and return top results.
    
    Args:
        client: Qdrant client instance
        vector: Query vector (list or array)
        collection_name: Name of the collection to query
        top_k: Number of results to return
        vector_name: Name of the vector field (for collections with named vectors)
                     If provided, the query_vector will be formatted as (vector_name, vector)
    
    Returns:
        list: Search results, or empty list if query fails
    """
    print(f"[INFO] Querying collection '{collection_name}' ...")
    
    try:
        # First, check collection info to diagnose potential issues
        try:
            collection_info = client.get_collection(collection_name)
            collection_size = collection_info.points_count
            print(f"[DEBUG] Collection '{collection_name}' info: {collection_size} points total")
            
            if collection_size == 0:
                print(f"[WARN] Collection '{collection_name}' is empty! No vectors to search.")
                return []
        except Exception as e:
            print(f"[WARN] Could not retrieve collection info: {e}")
        
        # Format query_vector: if vector_name is specified, pass as tuple (vector_name, vector)
        if vector_name is not None:
            query_vector = (vector_name, vector)
            print(f"[DEBUG] Using named vector: '{vector_name}'")
        else:
            query_vector = vector
        
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        if len(results) == 0:
            print(f"[WARN] Query returned 0 results from '{collection_name}'.")
            print(f"[HINT] Possible causes:")
            print(f"       1. No vectors in collection match the query vector")
            print(f"       2. Query vector dimension may not match collection schema")
            if vector_name:
                print(f"       3. Vector name '{vector_name}' might not exist in collection")
            print(f"       4. All matching results may be filtered out by score threshold")
            return []
        
        print(f"[OK] Retrieved {len(results)} results from '{collection_name}'.")
        
        # Log score range for diagnostics
        if results:
            scores = [r.score for r in results]
            print(f"[DEBUG] Score range: {min(scores):.4f} - {max(scores):.4f}")
        
        return results
    
    except (AssertionError, TypeError, ValueError) as e:
        # If named vector fails, try without it
        if vector_name is not None and ("Unknown arguments" in str(e) or "vector_name" in str(e)):
            print(f"[WARN] Named vector query failed, retrying with default vector...")
            try:
                results = client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=True
                )
                if len(results) == 0:
                    print(f"[WARN] Query returned 0 results from '{collection_name}' (fallback to default vector).")
                    print(f"[HINT] This might indicate:")
                    print(f"       1. Named vector '{vector_name}' exists but may not have relevant data")
                    print(f"       2. No vectors in collection match the query vector")
                else:
                    print(f"[OK] Retrieved {len(results)} results from '{collection_name}' (fallback to default vector).")
                    scores = [r.score for r in results]
                    print(f"[DEBUG] Score range: {min(scores):.4f} - {max(scores):.4f}")
                return results
            except Exception as e2:
                print(f"[ERROR] Query failed even with fallback: {type(e2).__name__}: {e2}")
                return []
        else:
            print(f"[ERROR] Query failed: {type(e).__name__}: {e}")
            return []
    
    except Exception as e:
        print(f"[ERROR] Unexpected error querying collection '{collection_name}': {type(e).__name__}: {e}")
        if "400" in str(e) or "Bad Request" in str(e):
            print(f"[HINT] This is typically a request format error. Check:")
            print(f"       1. Vector dimension matches collection schema")
            print(f"       2. Vector name is valid for the collection")
            print(f"       3. Request parameters are properly formatted")
        return []

def merge_results(text_results, image_results):
    """
    Merge text and image search results by common case_id.
    Returns a dict grouped by case_id, consistent with MAJOR_REFRACTOR.md.
    
    Note: Both text and image results include text summary fields in their payload.
    This function ensures text summaries are preserved from whichever source retrieves them.
    """
    print("[INFO] Merging Qdrant query results...")

    retrieved_cases = {}
    
    # Text summary field names to extract from payloads
    TEXT_FIELDS = {"Title", "History", "Clinical_Findings", "Discussion", "Summary_Box_First_Line"}

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
        
        # Extract text summary fields from image payload
        image_text_summary = {k: v for k, v in res.payload.items() if k in TEXT_FIELDS}
        
        if cid in retrieved_cases:
            # Case already exists from text results
            retrieved_cases[cid]["images"].append(res.payload)
        else:
            # case_id appears only in image results
            # Use text summary from image payload or create empty dict if not available
            retrieved_cases[cid] = {
                "case_id": cid,
                "text": image_text_summary if image_text_summary else {},
                "images": [res.payload]
            }
            
            if image_text_summary:
                print(f"[INFO] Text summary for case {cid} extracted from image result")

    print(f"[OK] Merged {len(retrieved_cases)} unique cases.")
    return retrieved_cases

def normalize_cases_for_output(retrieved_cases):
    """
    Normalize retrieved cases for final diagnostic output.
    Extracts only the specified fields per the plan: title, history, clinical_findings, 
    discussion, and summary_box_first_line (regardless of modality).
    Removes redundant image payloads and nested structures.
    
    Args:
        retrieved_cases: Dict of retrieved cases from merge_results
        
    Returns:
        dict: Normalized cases with only essential fields, keyed by case_id
    """
    normalized = {}
    
    # Field name mappings: database names -> output names (lowercase)
    FIELD_MAPPING = {
        "Title": "title",
        "History": "history",
        "Clinical_Findings": "clinical_findings",
        "Discussion": "discussion",
        "Summary_Box_First_Line": "summary_box_first_line"
    }
    
    for case_id, case_data in retrieved_cases.items():
        # Extract text summary (can come from either text or images)
        text_payload = case_data.get("text", {})
        
        # Create normalized case with only specified fields
        normalized_case = {}
        for db_field, output_field in FIELD_MAPPING.items():
            normalized_case[output_field] = text_payload.get(db_field, "")
        
        normalized[f"case_{case_id}"] = normalized_case
    
    print(f"[OK] Normalized {len(normalized)} cases for final output.")
    return normalized

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

def run_query(text_vector=None, image_vector=None, top_k=None, session_id=None):
    """
    Orchestrate Qdrant query for one or both modalities.
    Robust to individual modality failures - returns whatever results are available.
    
    Args:
        text_vector: numpy array or list for text query (optional)
        image_vector: numpy array or list for image query (optional)
        top_k: number of results to return (optional, uses config if not specified)
        session_id: session identifier (for temp data path)
    
    Returns:
        tuple: (retrieved_cases dict, saved_dir path) or (None, None) if no vectors provided
               Returns partial results if one modality fails
    """
    # Use config values if top_k not specified
    if top_k is None:
        # For consistency, use text_top_k as default; caller can specify if needed
        text_top_k = QUERY_CONFIG.get("text_top_k", 5)
        image_top_k = QUERY_CONFIG.get("image_top_k", 5)
    else:
        # If single top_k provided, use for both
        text_top_k = top_k
        image_top_k = top_k
    
    # Validate that at least one vector is provided
    if text_vector is None and image_vector is None:
        print("[WARN] No query vectors provided — nothing to search.")
        return None, None
    
    try:
        client = setup_qdrant_client()
    except Exception as e:
        print(f"[ERROR] Failed to connect to Qdrant: {e}")
        return None, None
    
    text_results = []
    image_results = []
    query_status = []

    # Query text modality if provided
    if text_vector is not None:
        print("[INFO] Text vector provided. Querying text collection...")
        try:
            text_results = query_collection(client, text_vector, COLLECTION_NAME_TEXT, text_top_k)
            if text_results:
                query_status.append("✓ Text query successful")
                print(f"[OK] Text query returned {len(text_results)} results (top_k={text_top_k}).")
            else:
                query_status.append("⚠ Text query returned no results")
                print("[WARN] Text query returned no results.")
        except Exception as e:
            query_status.append(f"✗ Text query failed: {e}")
            print(f"[ERROR] Text query failed: {e}")
            print("[INFO] Continuing with image query...")
    else:
        print("[INFO] No text vector provided. Skipping text search.")
    
    # Query image modality if provided
    if image_vector is not None:
        print("[INFO] Image vector provided. Querying image collection...")
        print(f"[DEBUG] Image vector shape/dimension: {len(image_vector) if isinstance(image_vector, list) else image_vector.shape}")
        try:
            # Image collection has named vectors, try with 'image' vector name
            image_results = query_collection(client, image_vector, COLLECTION_NAME_IMAGE, image_top_k, vector_name="image")
            if image_results:
                query_status.append("✓ Image query successful")
                print(f"[OK] Image query returned {len(image_results)} results (top_k={image_top_k}).")
            else:
                query_status.append("⚠ Image query returned no results")
                print("[WARN] Image query returned no results (see hints above for diagnostics).")
        except Exception as e:
            query_status.append(f"✗ Image query failed: {type(e).__name__}")
            print(f"[ERROR] Image query failed: {type(e).__name__}: {e}")
            print("[INFO] Continuing with text results...")
    else:
        print("[INFO] No image vector provided. Skipping image search.")

    # Check if we got ANY results
    if not text_results and not image_results:
        print(f"[WARN] No cases retrieved from any modality.")
        print(f"[INFO] Query status: {', '.join(query_status)}")
        return None, None
    
    # Log query status
    print(f"[INFO] Query summary: {', '.join(query_status)}")
    
    # Merge available results
    retrieved_cases = merge_results(text_results, image_results)
    
    if not retrieved_cases:
        print("[WARN] No cases after merging results.")
        return None, None
    
    try:
        saved_dir = save_retrieved_cases(retrieved_cases, session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(f"[OK] Query complete. Retrieved {len(retrieved_cases)} cases total from {len(query_status)} modalities.")
        return retrieved_cases, saved_dir
    except Exception as e:
        print(f"[ERROR] Failed to save retrieved cases: {e}")
        return None, None



if __name__ == "__main__":
    # Example run (for testing)
    import numpy as np
    dummy_vec = np.random.rand(768).tolist()  # Example vector
    run_query(text_vector=dummy_vec, top_k=3, session_id="test_session")
