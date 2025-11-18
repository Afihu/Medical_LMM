# Batch Evaluation

Automated RAGAS evaluation for Medical_LMM test cases with support for **Gemini** or **local LLMs** (LM Studio, vLLM, Ollama).

---

## Quick Start

### Using Gemini (Cloud)

```bash
# Set in .env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here

# Run evaluation
python evaluate/run_batch_evaluation.py --limit 5
```

### Using Local LLM (LM Studio)

```bash
# Set in .env
LLM_PROVIDER=lmstudio
LOCAL_LLM_URL=http://localhost:1234
LMSTUDIO_MODEL=medgemma-4b-it

# Run evaluation
python evaluate/run_batch_evaluation.py --limit 5
```

📖 **Full Guide**: See [`LOCAL_LLM_EVALUATION_GUIDE.md`](./LOCAL_LLM_EVALUATION_GUIDE.md) for detailed setup instructions.

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

## Requirements

```bash
pip install ragas langchain-google-genai langchain-huggingface qdrant-client sentence-transformers
```

Set `GEMINI_API_KEY` in `.env` file.

