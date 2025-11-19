# Batch Evaluation

Automated RAGAS evaluation for Medical_LMM test cases with support for **Gemini** or **local LLMs** (LM Studio, vLLM, Ollama).

---

## Quick Start

### Using Gemini (Cloud)

```bash
# Set in scripts\config\llm_config.py
LLM_PROVIDER=gemini
# Set in .env file
GEMINI_API_KEY=your_api_key_here

# Run evaluation
python evaluate/run_batch_evaluation.py --limit 5
# If you use uv
uv run python evaluate/run_batch_evaluation.py --limit 5
```

### Using Local LLM (LM Studio)

```bash
# Set in scripts\config\llm_config.py
LLM_PROVIDER=lmstudio
LOCAL_LLM_URL=http://localhost:1234
LMSTUDIO_MODEL=medgemma-4b-it

# Run evaluation
python evaluate/run_batch_evaluation.py --limit 5
# If you use uv
uv run python evaluate/run_batch_evaluation.py --limit 5
```
---

## Pipeline

1. Generate diagnosis for each test case (Qdrant retrieval + Gemini)
2. Evaluate with 5 RAGAS metrics:
   - **Context Precision** - Relevance of retrieved cases
   - **Context Recall** - Completeness of retrieved context
   - **Faithfulness** - Answer grounded in retrieved context
   - **Answer Relevancy** - Answer relevance to query
   - **Answer Correctness** - Diagnosis accuracy vs ground truth
3. Generate reports: JSON + CSV + Markdown

## Output

Results in `evaluate/batch_results/session_YYYYMMDD_HHMMSS/`:
- `evaluation_results.csv` - Tabular data for analysis
- `evaluation_results.json` - Full evaluation metadata
- `evaluation_report.md` - Human-readable report

## Options

```bash
--limit N           # Evaluate first N cases only
--skip N            # Skip first N cases
--skip-diagnosis    # Use existing diagnosis files (if available)
```

### In evaluate/eval_config.py
- You can change:
   - Evaluation mode (internal, rag, hybrid), this also affects which prompt is used for evaluation.
   - Context type (text, text+image)
   - RAGAS LLM provider and embedding model

- The other settings:
   - Diagnosis generation LLM in `scripts\config\llm_config.py`
   - Qdrant query settings in `scripts\config\query_config.py`
   - You can modify the specific prompts in `scripts\main_runtime`, including:
      - `prompt_internal.txt`
      - `prompt_rag.txt`
      - `prompt_hybrid.txt`

## Requirements

```bash
pip install ragas langchain-google-genai langchain-huggingface qdrant-client sentence-transformers
# Or with uv
uv sync
```

- Set `GEMINI_API_KEY` in `.env` file.

