"""
Query Configuration for Main Runtime Pipeline.

This module specifies the top-k values for retrieving similar cases from Qdrant collections.
These parameters can be tuned independently for each modality (text and image).
"""

# Top-k retrieval parameters for each modality
QUERY_CONFIG = {
    "text_top_k": 3,       # Number of top similar text cases to retrieve
    "image_top_k": 3,      # Number of top similar image cases to retrieve
}

# Collection names in Qdrant
QDRANT_COLLECTIONS = {
    "text": "medical_case_texts",
    "image": "medical_case_images",
}

# Query thresholds (optional - for filtering results by similarity score)
QUERY_THRESHOLDS = {
    "text_score_threshold": 0.0,   # Minimum similarity score for text (0.0 = no filtering)
    "image_score_threshold": 0.0,  # Minimum similarity score for image (0.0 = no filtering)
}
