"""
Evaluation Configuration
------------------------
Central configuration for batch evaluation scenarios with different modes and context types.

Usage:
    from evaluate.config import EVAL_MODE, CONTEXT_TYPE, get_config
    
    config = get_config()
    print(f"Mode: {config['mode']}, Context: {config['context_type']}")
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# EVALUATION MODES
# =============================================================================
# - "internal": LLM uses only internal knowledge, no RAG retrieval
# - "rag": LLM uses only retrieved context, no internal knowledge (RAG-only)
# - "hybrid": LLM uses both internal knowledge + retrieved context (default)
# =============================================================================
EVAL_MODE = os.getenv("EVAL_MODE", "hybrid").lower()

# =============================================================================
# CONTEXT TYPES
# =============================================================================
# - "text": Use only text context (default)
# - "text+images": Use both text and images if available
# =============================================================================
CONTEXT_TYPE = os.getenv("CONTEXT_TYPE", "text").lower()

# =============================================================================
# LLM PROVIDER CONFIGURATION
# =============================================================================
# Which LLM provider to use for diagnosis and evaluation
# - "gemini": Google Gemini (requires GEMINI_API_KEY)
# - "lmstudio": Local LM Studio or compatible server (requires LOCAL_LLM_URL)
# Defaults to "gemini" if not specified
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# For local LLM provider, specify the endpoint URL
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:1234")

# =============================================================================
# LM STUDIO GENERATION PARAMETERS
# =============================================================================
# Model name to use in LM Studio (e.g., "medgemma-4b-it")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "medgemma-4b-it")

# Temperature controls response creativity (0.2 = deterministic, 0.9 = creative)
LMSTUDIO_TEMPERATURE = float(os.getenv("LMSTUDIO_TEMPERATURE", "0.7"))

# Maximum tokens for generation (32K context window default)
LMSTUDIO_MAX_TOKENS = int(os.getenv("LMSTUDIO_MAX_TOKENS", "32768"))

# Top-p controls token probability sampling (0.95 balances diversity & coherence)
LMSTUDIO_TOP_P = float(os.getenv("LMSTUDIO_TOP_P", "0.95"))

# =============================================================================
# RAGAS EVALUATION CONFIGURATION
# =============================================================================
# Which LLM to use for RAGAS metrics evaluation
# - "gemini": Google Gemini (via langchain_google_genai)
# - "local": Use local LLM server (via LangChain HuggingFace adapter)
RAGAS_LLM_PROVIDER = os.getenv("RAGAS_LLM_PROVIDER", "gemini").lower()

# Which embeddings model to use for RAGAS answer_relevancy metric
# - "google": Google's embedding model (via GoogleGenerativeAIEmbeddings)
# - "huggingface": HuggingFace embeddings (via HuggingFaceEmbeddings)
# - "sentence-transformers": SentenceTransformers embeddings
RAGAS_EMBEDDINGS_PROVIDER = os.getenv("RAGAS_EMBEDDINGS_PROVIDER", "google").lower()

# Specific HuggingFace or SentenceTransformers model name (if using HF embeddings)
RAGAS_EMBEDDINGS_MODEL = os.getenv(
    "RAGAS_EMBEDDINGS_MODEL",
    "abhinand/MedEmbed-base-v0.1"  # Medical-specific embedding model
)

# =============================================================================
# QDRANT CONFIGURATION
# =============================================================================
# Retrieved context top_k
QDRANT_TEXT_TOP_K = int(os.getenv("QDRANT_TEXT_TOP_K", "3"))
QDRANT_IMAGE_TOP_K = int(os.getenv("QDRANT_IMAGE_TOP_K", "3"))

# =============================================================================
# VALIDATION & DEFAULTS
# =============================================================================
def validate_config():
    """Validate configuration and raise errors for invalid settings."""
    valid_modes = {"internal", "rag", "hybrid"}
    valid_contexts = {"text", "text+images"}
    valid_providers = {"gemini", "lmstudio", "local"}
    
    if EVAL_MODE not in valid_modes:
        raise ValueError(f"EVAL_MODE must be one of {valid_modes}, got '{EVAL_MODE}'")
    
    if CONTEXT_TYPE not in valid_contexts:
        raise ValueError(f"CONTEXT_TYPE must be one of {valid_contexts}, got '{CONTEXT_TYPE}'")
    
    if LLM_PROVIDER not in valid_providers:
        raise ValueError(f"LLM_PROVIDER must be one of {valid_providers}, got '{LLM_PROVIDER}'")
    
    if RAGAS_LLM_PROVIDER not in {"gemini", "local"}:
        raise ValueError(f"RAGAS_LLM_PROVIDER must be 'gemini' or 'local', got '{RAGAS_LLM_PROVIDER}'")
    
    if RAGAS_EMBEDDINGS_PROVIDER not in {"google", "huggingface", "sentence-transformers"}:
        raise ValueError(
            f"RAGAS_EMBEDDINGS_PROVIDER must be 'google', 'huggingface', or 'sentence-transformers', "
            f"got '{RAGAS_EMBEDDINGS_PROVIDER}'"
        )
    
    # Check required environment variables for chosen providers
    if LLM_PROVIDER == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise ValueError("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set in .env")
    
    if LLM_PROVIDER == "lmstudio" and not os.getenv("LOCAL_LLM_URL"):
        raise ValueError("LLM_PROVIDER is 'lmstudio' but LOCAL_LLM_URL is not set in .env")
    
    if RAGAS_LLM_PROVIDER == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise ValueError("RAGAS_LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set in .env")


def get_config():
    """
    Get the complete configuration dictionary.
    
    Returns:
        dict: Configuration with all settings
    """
    validate_config()
    
    return {
        "mode": EVAL_MODE,
        "context_type": CONTEXT_TYPE,
        "llm_provider": LLM_PROVIDER,
        "local_llm_url": LOCAL_LLM_URL,
        "lmstudio_model": LMSTUDIO_MODEL,
        "lmstudio_temperature": LMSTUDIO_TEMPERATURE,
        "lmstudio_max_tokens": LMSTUDIO_MAX_TOKENS,
        "lmstudio_top_p": LMSTUDIO_TOP_P,
        "ragas_llm_provider": RAGAS_LLM_PROVIDER,
        "ragas_embeddings_provider": RAGAS_EMBEDDINGS_PROVIDER,
        "ragas_embeddings_model": RAGAS_EMBEDDINGS_MODEL,
        "qdrant_text_top_k": QDRANT_TEXT_TOP_K,
        "qdrant_image_top_k": QDRANT_IMAGE_TOP_K,
    }


def print_config():
    """Pretty-print the current configuration."""
    config = get_config()
    print("\n" + "=" * 60)
    print("📋 EVALUATION CONFIGURATION")
    print("=" * 60)
    print(f"  Mode:                    {config['mode'].upper()}")
    print(f"  Context Type:            {config['context_type'].upper()}")
    print(f"  LLM Provider:            {config['llm_provider'].upper()}")
    if config['llm_provider'] == 'lmstudio':
        print(f"  Local LLM URL:           {config['local_llm_url']}")
        print(f"  LM Studio Model:         {config['lmstudio_model']}")
        print(f"  Temperature:             {config['lmstudio_temperature']}")
        print(f"  Max Tokens:              {config['lmstudio_max_tokens']}")
        print(f"  Top-P:                   {config['lmstudio_top_p']}")
    print(f"  RAGAS LLM:               {config['ragas_llm_provider'].upper()}")
    print(f"  RAGAS Embeddings:        {config['ragas_embeddings_provider'].upper()}")
    if config['ragas_embeddings_provider'] != 'google':
        print(f"  Embeddings Model:        {config['ragas_embeddings_model']}")
    print(f"  Qdrant Text Top-K:       {config['qdrant_text_top_k']}")
    print(f"  Qdrant Image Top-K:      {config['qdrant_image_top_k']}")
    print("=" * 60 + "\n")


# =============================================================================
# MODE DESCRIPTIONS
# =============================================================================
MODE_DESCRIPTIONS = {
    "internal": """
    INTERNAL KNOWLEDGE ONLY
    - LLM generates diagnosis using only internal knowledge
    - No Qdrant retrieval (RAG disabled)
    - Useful for: Testing LLM reasoning without external context
    - Diagnosis generation is faster but may be less accurate
    """,
    "rag": """
    RAG ONLY (Retrieval-Augmented Generation)
    - LLM uses only retrieved context from Qdrant
    - Prompt explicitly hides internal knowledge instruction
    - Useful for: Testing retrieval quality and context relevance
    - Diagnosis depends entirely on retrieved cases quality
    """,
    "hybrid": """
    HYBRID (Default)
    - LLM uses both internal knowledge AND retrieved context
    - Qdrant retrieval is enabled
    - Prompt encourages using context but allows internal reasoning
    - Useful for: Production usage, best of both worlds
    """,
}

CONTEXT_DESCRIPTIONS = {
    "text": """
    TEXT ONLY
    - Use only text-based context (clinical history, findings, etc.)
    - Faster evaluation, simpler retrieval
    - Useful for: Text-only medical cases or quick testing
    """,
    "text+images": """
    TEXT + IMAGES
    - Use both text context and medical images if available
    - Slower evaluation (image embedding required)
    - Requires CLIP or vision model for image embeddings
    - Useful for: Comprehensive evaluation with visual data
    """,
}


if __name__ == "__main__":
    # Test configuration when run directly
    try:
        print_config()
        print("✅ Configuration is valid!")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
