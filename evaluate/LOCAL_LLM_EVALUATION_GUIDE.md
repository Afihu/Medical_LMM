# Local LLM Evaluation Guide
**Phase 2 & 3 Implementation - Complete**

---

## Overview

The evaluation pipeline now supports **local LLM servers** (LM Studio, vLLM, Ollama) for both diagnosis generation and RAGAS metrics evaluation. You can choose between cloud-based (Gemini) or local inference for full control and cost savings.

---

## Quick Start

### Option 1: Gemini (Cloud - Default)

```bash
# .env configuration
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
RAGAS_LLM_PROVIDER=gemini
RAGAS_EMBEDDINGS_PROVIDER=google

# Run evaluation
python evaluate/run_batch_evaluation.py --limit 5
```

### Option 2: Local LLM (LM Studio/vLLM/Ollama)

```bash
# .env configuration
LLM_PROVIDER=lmstudio
LOCAL_LLM_URL=http://localhost:1234
LMSTUDIO_MODEL=medgemma-4b-it

# Use local LLM for RAGAS metrics too
RAGAS_LLM_PROVIDER=local
RAGAS_EMBEDDINGS_PROVIDER=huggingface
RAGAS_EMBEDDINGS_MODEL=abhinand/MedEmbed-base-v0.1

# Run evaluation
python evaluate/run_batch_evaluation.py --limit 5
```

### Option 3: Hybrid (Local diagnosis + Gemini metrics)

```bash
# .env configuration
LLM_PROVIDER=lmstudio              # Local LLM for diagnosis
LOCAL_LLM_URL=http://localhost:1234
LMSTUDIO_MODEL=medgemma-4b-it

RAGAS_LLM_PROVIDER=gemini          # Gemini for evaluation metrics
GEMINI_API_KEY=your_api_key_here
RAGAS_EMBEDDINGS_PROVIDER=google

# Run evaluation
python evaluate/run_batch_evaluation.py --limit 5
```

---

## Configuration Reference

### LLM Provider Settings

| Variable | Options | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini`, `lmstudio` | Which LLM to use for diagnosis generation |
| `LOCAL_LLM_URL` | `http://localhost:1234` | OpenAI-compatible endpoint URL |
| `LMSTUDIO_MODEL` | `medgemma-4b-it` | Model name in LM Studio |
| `LMSTUDIO_TEMPERATURE` | `0.7` | Generation temperature (0.0-1.0) |
| `LMSTUDIO_MAX_TOKENS` | `32768` | Max tokens for generation |
| `LMSTUDIO_TOP_P` | `0.95` | Top-p sampling parameter |

### RAGAS Evaluation Settings

| Variable | Options | Description |
|----------|---------|-------------|
| `RAGAS_LLM_PROVIDER` | `gemini`, `local` | Which LLM to use for RAGAS metrics |
| `RAGAS_EMBEDDINGS_PROVIDER` | `google`, `huggingface` | Embeddings model provider |
| `RAGAS_EMBEDDINGS_MODEL` | `abhinand/MedEmbed-base-v0.1` | Specific embeddings model name |

### Evaluation Mode Settings

| Variable | Options | Description |
|----------|---------|-------------|
| `EVAL_MODE` | `internal`, `rag`, `hybrid` | Evaluation mode (default: `hybrid`) |
| `CONTEXT_TYPE` | `text`, `text+images` | Context type (default: `text`) |
| `QDRANT_TEXT_TOP_K` | `3` | Number of text contexts to retrieve |
| `QDRANT_IMAGE_TOP_K` | `3` | Number of image contexts to retrieve |

---

## Supported Local LLM Servers

### 1. LM Studio
- **Installation**: https://lmstudio.ai/
- **Default URL**: `http://localhost:1234`
- **Models**: Any GGUF or HuggingFace model
- **Recommended**: `medgemma-4b-it`, `llama-3.1-8b-instruct`

### 2. vLLM
- **Installation**: `pip install vllm`
- **Start server**: 
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model google/medgemma-4b-it \
    --port 1234
  ```
- **URL**: `http://localhost:1234`

### 3. Ollama
- **Installation**: https://ollama.ai/
- **Start server**: `ollama serve`
- **Default URL**: `http://localhost:11434`
- **Models**: `ollama pull medgemma:4b`

---

## Testing

### Test Local LLM Support

```bash
# Test with Gemini
python evaluate/test_local_llm_eval.py --provider gemini

# Test with local LLM
python evaluate/test_local_llm_eval.py --provider local

# Test both
python evaluate/test_local_llm_eval.py --provider both
```

---

## Architecture

### Diagnosis Generation Flow

```
User Query
    ↓
LLMProvider (via factory)
    ├─→ GeminiProvider (if LLM_PROVIDER=gemini)
    └─→ LMStudioProvider (if LLM_PROVIDER=lmstudio)
    ↓
DiagnosisRunner.run()
    ↓
Diagnosed Case JSON
```

### RAGAS Evaluation Flow

```
Diagnosed Case JSON
    ↓
RAGASEvaluator.evaluate_case()
    ↓
RAGASEvaluator._evaluate_in_subprocess()
    ↓
ragas_worker.py (subprocess)
    ├─→ ChatGoogleGenerativeAI (if RAGAS_LLM_PROVIDER=gemini)
    └─→ ChatOpenAI + base_url (if RAGAS_LLM_PROVIDER=local)
    ↓
    ├─→ GoogleGenerativeAIEmbeddings (if RAGAS_EMBEDDINGS_PROVIDER=google)
    └─→ HuggingFaceEmbeddings (if RAGAS_EMBEDDINGS_PROVIDER=huggingface)
    ↓
RAGAS Metrics (context_precision, context_recall, faithfulness, etc.)
```

---

## Common Issues & Solutions

### 1. "Connection refused" error

**Problem**: Local LLM server not running

**Solution**:
```bash
# Check if LM Studio is running
curl http://localhost:1234/v1/models

# Start LM Studio and load a model
# Or start vLLM/Ollama server
```

### 2. "API key not found" error

**Problem**: Using Gemini but API key missing

**Solution**:
```bash
# Add to .env
GEMINI_API_KEY=your_api_key_here
```

### 3. Slow evaluation with local LLM

**Problem**: Local LLM inference is slower than Gemini

**Solutions**:
- Use smaller models (4B instead of 13B)
- Enable GPU acceleration in LM Studio
- Reduce `LMSTUDIO_MAX_TOKENS` to 4096 for faster responses
- Use vLLM for optimized inference

### 4. "Model not found" error

**Problem**: Model name mismatch between config and LM Studio

**Solution**:
```bash
# Check available models in LM Studio
curl http://localhost:1234/v1/models

# Update .env with exact model name
LMSTUDIO_MODEL=exact-model-name-from-lm-studio
```

---

## Performance Comparison

| Provider | Diagnosis Speed | RAGAS Speed | Cost | Quality |
|----------|----------------|-------------|------|---------|
| **Gemini** | ⚡ Fast | ⚡ Fast | 💰 Paid | 🌟🌟🌟🌟🌟 |
| **Local 4B** | 🐌 Slow | 🐌 Slow | ✅ Free | 🌟🌟🌟 |
| **Local 8B** | 🐌 Slower | 🐌 Slower | ✅ Free | 🌟🌟🌟🌟 |
| **Hybrid** | ⚡/🐌 Mixed | ⚡ Fast | 💰 Half | 🌟🌟🌟🌟 |

**Recommendation**: Use **Hybrid** mode for cost-effective evaluation with high-quality metrics.

---

## Advanced Usage

### Custom Embeddings Model

```bash
# Use different medical embeddings
RAGAS_EMBEDDINGS_PROVIDER=huggingface
RAGAS_EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Multiple Evaluation Runs

```bash
# Gemini run
LLM_PROVIDER=gemini python evaluate/run_batch_evaluation.py --limit 10

# Local LLM run
LLM_PROVIDER=lmstudio python evaluate/run_batch_evaluation.py --limit 10

# Compare results
python evaluate/utils/report_generator.py --compare session_1 session_2
```

---

## Implementation Details

### Files Modified (Phase 2 & 3)

| File | Changes |
|------|---------|
| `evaluate/utils/ragas_worker.py` | Added local LLM support via `ChatOpenAI` with custom `base_url` |
| `evaluate/utils/ragas_evaluator.py` | Pass RAGAS config to subprocess worker |
| `evaluate/config.py` | Already supported all providers (no changes needed) |
| `evaluate/test_local_llm_eval.py` | New test script for validation |

### Key Features

✅ **Provider-agnostic RAGAS evaluation**
- Supports Gemini and local LLMs
- Uses LangChain `ChatOpenAI` with custom `base_url`
- Compatible with LM Studio, vLLM, Ollama

✅ **Embeddings provider selection**
- Google embeddings (via `GoogleGenerativeAIEmbeddings`)
- HuggingFace embeddings (via `HuggingFaceEmbeddings`)
- Medical-specific model support (`abhinand/MedEmbed-base-v0.1`)

✅ **Backward compatibility**
- Existing Gemini-only code still works
- No breaking changes to API
- Configuration-driven, easy to switch

---

## Next Steps

1. **Run tests** to validate your setup:
   ```bash
   python evaluate/test_local_llm_eval.py --provider both
   ```

2. **Start small** with a few test cases:
   ```bash
   python evaluate/run_batch_evaluation.py --limit 5
   ```

3. **Scale up** for full evaluation:
   ```bash
   python evaluate/run_batch_evaluation.py --limit 50
   ```

4. **Compare providers** and choose the best for your needs

---

## Support

For issues or questions:
- Check the console output for detailed error messages
- Review `evaluate/config.py` for configuration options
- Consult `BATCH_EVAL_README.md` for general evaluation guide
- See memory files for implementation details

---

**Status**: ✅ Phase 2 & 3 Complete - Local LLM evaluation fully supported!
