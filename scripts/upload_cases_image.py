"""
upload_image_cases.py
---------------------
Uploads only image vectors from /temp_data into Qdrant Cloud.

Each JSON file in /temp_data should contain:
[
  {
    "id": "case_001",
    "vector": { "image_vector": [...] },
    "payload": {}
  },
  ...
]
"""

import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DATA_FOLDER = os.path.join(PROJECT_ROOT, "temp_data")

# --- Load credentials ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def setup_qdrant_client():
    """Connect to Qdrant Cloud."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in .env")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"Connected to Qdrant Cloud at {QDRANT_URL}\n")
    return client


def ensure_image_collection(client, collection_name, vector_size):
    """Ensure a collection for image vectors exists."""
    collections = [col.name for col in client.get_collections().collections]
    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists.\n")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"Collection '{collection_name}' created for image vectors (size={vector_size}).\n")


def load_cases():
    """Load case data containing image vectors."""
    if not os.path.exists(TEMP_DATA_FOLDER):
        raise FileNotFoundError(f"Missing folder: {TEMP_DATA_FOLDER}")

    files = [f for f in os.listdir(TEMP_DATA_FOLDER) if f.endswith(".json")]
    if not files:
        print("No case JSON files found in /temp_data.")
        return []

    all_cases = []
    for filename in files:
        path = os.path.join(TEMP_DATA_FOLDER, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_cases.extend(data)
            elif isinstance(data, dict):
                all_cases.append(data)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Loaded {len(all_cases)} cases from {len(files)} files.\n")
    return all_cases


def upload_image_cases():
    """Upload image-only cases."""
    client = setup_qdrant_client()
    collection_name = input("Enter collection name for IMAGE vectors: ").strip()

    all_cases = load_cases()
    if not all_cases:
        print("No valid cases to upload.")
        return

    # Detect vector size automatically
    first_vector = next(
        (c["vector"].get("image_vector") for c in all_cases if "vector" in c and "image_vector" in c["vector"]),
        None,
    )
    if first_vector is None:
        print("No 'image_vector' found in data.")   # So that means there are wrong data types
        return

    ensure_image_collection(client, collection_name, len(first_vector))

    points = []
    for case in all_cases:
        try:
            if "image_vector" not in case["vector"]:
                continue
            point = models.PointStruct(
                id=case["id"],
                vector=case["vector"]["image_vector"],
                payload=case.get("payload", {}),
            )
            points.append(point)
        except Exception as e:
            print(f"Skipped a case due to error: {e}")

    if not points:
        print("No valid image points to upload.")
        return

    # Batch upload to prevent connection reset
    BATCH_SIZE = 10
    print(f"Uploading {len(points)} points in batches of {BATCH_SIZE}...\n")

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        try:
            client.upsert(collection_name=collection_name, points=batch)
            print(f"Uploaded batch {i//BATCH_SIZE + 1} ({len(batch)} points)")
        except Exception as e:
            print(f"Failed batch {i//BATCH_SIZE + 1}: {e}")

    print("\nUpload complete. Exiting.\n")


if __name__ == "__main__":
    upload_image_cases()
