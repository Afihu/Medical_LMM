"""
[DEPRECATED - PLEASE REFER TO upload_cases_image.py and upload_cases_text.py]
upload_cases.py
---------------
Uploads case data from /temp_data to Qdrant Cloud.

Each JSON file in /temp_data should contain a list of case objects like:
[
  {
    "id": "case_001",
    "vector": {
      "text_vector": [...],
      "image_vector": [...]
    },
    "payload": { "diagnosis": "...", "symptoms": "...", "history": "..." }
  },
  ...
]

If the specified collection does not exist, the script will ask
for vector names and dimensions to create it.
"""

import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# --- Path setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DATA_FOLDER = os.path.join(PROJECT_ROOT, "temp_data")

# --- Load credentials ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# -------------------------------------------------------------------
def setup_qdrant_client():
    """Connect to Qdrant Cloud."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in .env")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"Connected to Qdrant Cloud at {QDRANT_URL}\n")
    return client


def ensure_collection(client, collection_name):
    """Check if collection exists; if not, ask for creation config."""
    collections = [col.name for col in client.get_collections().collections]
    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists.\n")
        return

    print(f"Collection '{collection_name}' does not exist.")
    print("Entering the new configuration:...")
    vector_config = {}

    while True:
        vec_name = input("Enter vector name (e.g., text_vector): ").strip()
        vec_dim = input(f"Enter dimension size for '{vec_name}': ").strip()
        if not vec_name or not vec_dim.isdigit():
            print("Invalid input. Please try again.")
            continue

        vector_config[vec_name] = models.VectorParams(
            size=int(vec_dim),
            distance=models.Distance.COSINE,
        )

        more = input("Add another vector? (y/n): ").strip().lower()
        if more != "y":
            break

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vector_config,
    )
    print(f"\n Collection '{collection_name}' created successfully!\n")


def load_case_files():
    """Load all case data from /temp_data folder."""
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
            else:
                print(f"Unsupported JSON format in {filename}. Skipping.")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Loaded {len(all_cases)} total cases from {len(files)} files.\n")
    return all_cases


def upload_cases():
    """Main uploader logic."""
    client = setup_qdrant_client()

    # Step 1: Ask for collection name
    collection_name = input("Enter target collection name: ").strip()
    ensure_collection(client, collection_name)

    # Step 2: Load and prepare points
    all_cases = load_case_files()
    if not all_cases:
        print("No valid cases to upload. Exiting.")
        return

    points = []
    for case in all_cases:
        try:
            point = models.PointStruct(
                id=case["id"],
                vector=case["vector"],
                payload=case.get("payload", {}),
            )
            points.append(point)
        except Exception as e:
            print(f"Skipped a case due to error: {e}")

    # Step 3: Upload
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded {len(points)} cases to '{collection_name}' successfully.")
    else:
        print("No valid points to upload.")

    print("\nUpload complete. Exiting program.\n")


# --- Entry Point ---
if __name__ == "__main__":
    upload_cases()
