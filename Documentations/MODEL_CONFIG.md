# Model Configuration

**All models configured in**: `scripts/config/model_config.py`

Edit `scripts/config/model_config.py`:
```python
DIAGNOSIS_MODEL = "gemini-2.5-flash"           # Generates diagnosis
RAGAS_EVALUATION_MODEL = "gemini-2.5-flash"    # Evaluates quality
RAGAS_EMBEDDINGS_MODEL = "models/text-embedding-004"  # Embeddings only
```

