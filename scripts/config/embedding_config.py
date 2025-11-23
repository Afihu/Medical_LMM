"""
Embedding Dimension Configuration
Standardizes embedding dimensions across all modules for consistency and validation.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONFIG = {
    "text_embedding_dim": 768,
    "image_embedding_dim": 1152,
    "verify_on_first_run": True,  # Test and alert if mismatch
    "image_resize_dim": 448,  # MedSigLip-448 requirement
    "model_cache_enabled": True,  # Use singleton pattern for models
    "sequential_processing": True,  # Load models sequentially to optimize memory usage
    "temp_data_cleanup": True,  # Auto cleanup in QueryOrchestrator
    "ragas_enabled": True,  # Enable RAGAS evaluation
    "huggingface_token": os.getenv("HUGGINGFACE_TOKEN"),  # For gated models like medsiglip-448
}

# Paths for data pipeline
PATHS = {
    "source_materials": "source_materials",
    "extracted_data": {
        "root": "extracted_data",
        "text": "extracted_data/text",
        "images": "extracted_data/images",
        "captions": "extracted_data/captions",
    },
    "staged_embeddings": {
        "root": "staged_embeddings",
        "text": "staged_embeddings/text_embeddings",
        "images": "staged_embeddings/image_embeddings",
        "captions": "staged_embeddings/caption_embeddings",
    },
}

# Naming conventions
NAMING_CONVENTIONS = {
    "text": "case_{case_id:03d}_text.json",
    "image": "case_{case_id:03d}_image_{image_id:03d}.png",
    "caption": "case_{case_id:03d}_caption_{caption_id:03d}.json",
    "text_embedding": "case_{case_id:03d}_text_embedding.npy",
    "image_embedding": "case_{case_id:03d}_image_{image_id:03d}_embedding.npy",
    "caption_embedding": "case_{case_id:03d}_caption_{caption_id:03d}_embedding.npy",
}

# Target Qdrant instance and collection names
QDRANT_CONFIG = {
    "url": os.getenv("QDRANT_URL"),
    "api_key": os.getenv("QDRANT_API_KEY"),
    "collections": {
        "text_embeddings": "medical_case_texts",
        "image_embeddings": "medical_case_images",
    }
}