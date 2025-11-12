"""
Centralized Model Configuration
--------------------------------
Single source of truth for all AI models used in Medical_LMM project.

To change models, simply edit the values below.
"""

# ============================================================================
# DIAGNOSIS GENERATION MODEL
# ============================================================================
# Used by: DiagnosisRunner, Streamlit UI, Batch Evaluation
# Purpose: Analyzes patient symptoms and generates diagnosis
DIAGNOSIS_MODEL = "gemini-2.5-flash"


# ============================================================================
# RAGAS EVALUATION MODEL
# ============================================================================
# Used by: RAGASEvaluator, ragas_worker
# Purpose: Evaluates diagnosis quality (5 metrics: context precision/recall, 
#          faithfulness, answer relevancy, answer correctness)
RAGAS_EVALUATION_MODEL = "gemini-2.5-flash"


# ============================================================================
# RAGAS EMBEDDINGS MODEL
# ============================================================================
# Used by: ragas_worker (answer_relevancy metric only)
# Purpose: Creates embeddings for semantic similarity comparison
# Note: This is Google's embedding model, not a Gemini LLM
RAGAS_EMBEDDINGS_MODEL = "models/text-embedding-004"
