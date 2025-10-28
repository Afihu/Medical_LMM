"""
query.py
--------
This module connects to Qdrant Cloud, performs similarity search
using a given embedding vector (from CLIP, for example),
and saves the top 5 most similar results to /cases folder as JSON files.

"""

import os
import json
import shutil
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# --- Path configuration ---
# Get the project root (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # as this will return /scripts, so we wrap in another one
CASES_FOLDER_TEXT = os.path.join(PROJECT_ROOT, "cases_text")
CASES_FOLDER_IMAGE = os.path.join(PROJECT_ROOT, "cases_image")

# --- Load credentials ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME_TEXT = "medical_case_summaries_qdrant"  # Change to your Qdrant collection name
COLLECTION_NAME_IMAGE = "star_charts"

# -------------------------------------------------------------------

def setup_qdrant_client():
    """Connect to Qdrant Cloud instance."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"Connected to Qdrant Cloud at {QDRANT_URL}")
    return client

def query_similar_cases(client, query_vector, limit=5, mode = 'text'):
    """Perform similarity search and return top results."""
    if mode == 'text':
        results = client.search(
            collection_name=COLLECTION_NAME_TEXT,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return results
    else:
        results = client.search(
            collection_name=COLLECTION_NAME_IMAGE,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return results

def save_results(results, mode):
    """Save top cases to the /cases directory."""
    # Always clear old results for cleanliness
    if mode == 'text':
        if os.path.exists(CASES_FOLDER_TEXT):
            shutil.rmtree(CASES_FOLDER_TEXT)
        os.makedirs(CASES_FOLDER_TEXT)

        for i, hit in enumerate(results):
            case_data = {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            filename = os.path.join(CASES_FOLDER_TEXT, f"case_{i+1}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(case_data, f, indent=4, ensure_ascii=False)

        print(f"Saved {len(results)} similar cases to '{CASES_FOLDER_TEXT}/'")
    else:
        if os.path.exists(CASES_FOLDER_IMAGE):
            shutil.rmtree(CASES_FOLDER_IMAGE)
        os.makedirs(CASES_FOLDER_IMAGE)

        for i, hit in enumerate(results):
            case_data = {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            filename = os.path.join(CASES_FOLDER_IMAGE, f"case_{i+1}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(case_data, f, indent=4, ensure_ascii=False)

        print(f"Saved {len(results)} similar cases to '{CASES_FOLDER_IMAGE}/'")

def run_query(query_vector, mode):
    """Run the full similarity search process."""
    if mode == 'text':
        client = setup_qdrant_client()
        print("Querying Qdrant for similar cases (text)...")
        results = query_similar_cases(client, query_vector=query_vector, limit=5, mode = 'text')
        print("Query done!")
        save_results(results, mode = 'text')
    else:
        client = setup_qdrant_client()
        print("Querying Qdrant for similar cases (image)...")
        results = query_similar_cases(client, query_vector=query_vector, limit=5, mode = 'image')
        print("Query done!")
        save_results(results, mode = 'image')



if __name__ == "__main__":
    # Example run (for testing)
    test_vector = [0.12, -0.45, 0.78, 0.66]
    run_query(test_vector)
