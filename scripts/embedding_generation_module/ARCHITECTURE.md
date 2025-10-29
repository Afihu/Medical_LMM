"""
Embedding Generation Module - Architecture Summary

This document explains the refactored Embedding Generation Module structure,
which now supports both the Embedding Pipeline and Main Pipeline.

## Directory Structure

scripts/embedding_generation_module/
├── __init__.py
├── generators/
│   ├── __init__.py
│   ├── text_embedding_generator.py     # Text embedding logic
│   └── image_embedding_generator.py    # Image embedding logic
└── orchestrators/
    ├── __init__.py
    ├── embedding_orchestrator.py       # Embedding Pipeline orchestrator
    └── query_orchestrator.py           # Main Pipeline orchestrator


## Key Components

### 1. TextEmbeddingGenerator (generators/text_embedding_generator.py)
Pure embedding generation logic for text.

Features:
- Loads text embedding model (PubMedBERT) via ModelManager
- Generates embeddings from extracted text sections
- Verifies embedding dimensions
- Saves embeddings to specified location

Usage:
    generator = TextEmbeddingGenerator(model_name="...")
    embedding = generator.generate_embedding(text_sections)
    generator.generate_and_save_embedding(text_path, case_id)

### 2. ImageEmbeddingGenerator (generators/image_embedding_generator.py)
Pure embedding generation logic for images.

Features:
- Loads image embedding model (MedSigLip) via ModelManager
- Generates embeddings from images and captions
- Verifies embedding dimensions
- Saves embeddings to specified location

Usage:
    generator = ImageEmbeddingGenerator(model_name="...")
    embedding = generator.generate_embedding(image_path, caption_text)
    generator.generate_and_save_embedding(image_path, caption, case_id, image_id)

### 3. ModelManager (orchestrators/embedding_orchestrator.py)
Singleton pattern for managing model instances.

Features:
- Loads models once and reuses them
- Shared across both pipelines
- Reduces memory footprint and initialization overhead
- Can clear models if needed

Usage:
    text_gen = ModelManager.get_text_generator(model_name="...")
    image_gen = ModelManager.get_image_generator(model_name="...")
    ModelManager.clear_models()

### 4. EmbeddingOrchestrator (orchestrators/embedding_orchestrator.py)
Orchestrator for the Embedding Pipeline.

Features:
- Coordinates text and image embedding generation
- Handles batch processing of extracted data
- Stages embeddings to staged_embeddings/
- Provides comprehensive summaries

Usage:
    orchestrator = EmbeddingOrchestrator(staging_mode="production")
    results = orchestrator.generate_all_embeddings()

### 5. QueryOrchestrator (orchestrators/query_orchestrator.py)
Orchestrator for the Main Pipeline (query handling).

Features:
- Reuses embedding generators via ModelManager
- Embeds user queries (text or image)
- Stages embeddings to temp_query_data/session_id/
- Automatic cleanup with context manager

Usage:
    with QueryOrchestrator(session_id="user_session_123") as orchestrator:
        text_embedding = orchestrator.embed_text_query(query_text)
        image_embedding = orchestrator.embed_image_query(image_path, caption)
    # Auto cleanup on exit


## Workflow

### Embedding Pipeline (Production Batch Processing)
1. Data Extraction Module extracts text, images, captions
   └─ Output: extracted_data/text/, extracted_data/images/, extracted_data/captions/

2. EmbeddingOrchestrator coordinates embedding generation
   ├─ TextEmbeddingGenerator processes text files
   ├─ ImageEmbeddingGenerator processes image files
   └─ Both use ModelManager for efficient model loading

3. Embeddings staged to staged_embeddings/
   └─ Output: case_001_text_embedding.npy, case_001_image_001_embedding.npy

4. Qdrant Upload Service uploads embeddings to Qdrant

### Main Pipeline (Query Runtime Processing)
1. User provides query (text or image) via Streamlit app

2. QueryOrchestrator processes query
   ├─ Reuses embedding generators via ModelManager
   └─ No redundant model loading

3. Embeddings staged temporarily to temp_query_data/session_id/

4. Qdrant Query Module searches using query embeddings

5. Results returned to user

6. Session cleanup (auto via context manager)


## Model Loading Strategy

ModelManager implements singleton pattern:

```
First Query:
ModelManager.get_text_generator()
└─ Loads PubMedBERT model into memory

Subsequent Queries:
ModelManager.get_text_generator()
└─ Returns cached model (no reload)

Result:
- Embedding Pipeline: Models loaded once, used for batch processing
- Main Pipeline: Models loaded on first query, reused for all user queries
- Memory efficient: No model duplication across pipelines
```

## Staging Modes

### Production Mode (Embedding Pipeline)
- Output Directory: staged_embeddings/
- Persistence: Permanent (for Qdrant upload)
- Cleanup: Never

### Temporary Mode (Main Pipeline)
- Output Directory: temp_query_data/session_id/
- Persistence: Session duration
- Cleanup: Auto (via context manager) or manual


## Configuration

See embedding_config.py:

```python
CONFIG = {
    "text_embedding_dim": 768,
    "image_embedding_dim": 1024,
    "verify_on_first_run": True,      # Verify dimensions on init
    "model_cache_enabled": True,      # Use ModelManager singleton
    "temp_data_cleanup": True,        # Auto cleanup in QueryOrchestrator
    "ragas_enabled": True,            # Enable RAGAS evaluation
}
```

## Integration with RAGAS

Both orchestrators save embeddings that can be used for RAGAS evaluation:

```
QueryOrchestrator (with RAGAS):
1. Embed user query
2. Search Qdrant -> get top-k results
3. Get LLM response
4. RAGAS evaluator:
   - Uses same embedding generator (via ModelManager)
   - Computes metrics: faithfulness, context_relevance, answer_relevance
   - All in same embedding space
```

## Error Handling

Both generators and orchestrators:
- Verify embedding dimensions on initialization
- Handle missing input files gracefully
- Provide detailed error messages
- Continue processing on partial failures
- Report summary statistics


## Example Usage

### Embedding Pipeline (Batch)
```python
from scripts.embedding_generation_module.orchestrators import EmbeddingOrchestrator

orchestrator = EmbeddingOrchestrator(staging_mode="production")
results = orchestrator.generate_all_embeddings()

# Results contain:
# - Text embeddings: staged_embeddings/case_001_text_embedding.npy
# - Image embeddings: staged_embeddings/case_001_image_001_embedding.npy
```

### Main Pipeline (Query)
```python
from scripts.embedding_generation_module.orchestrators import QueryOrchestrator

with QueryOrchestrator(session_id="user_123") as orchestrator:
    # Embed user query
    text_emb = orchestrator.embed_text_query("Symptoms of fever?")
    
    # Search Qdrant with text_emb
    results = qdrant.search(text_emb)
    
    # Get LLM response
    llm_response = gemini_api.generate(results, query)
    
    # RAGAS evaluation (uses same embedding generator)
    metrics = ragas_evaluator.evaluate(query, results, llm_response)

# Auto cleanup on exit
```
"""
