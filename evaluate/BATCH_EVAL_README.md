# Batch Evaluation

Automated RAGAS evaluation for 186 Medical_LMM test cases.

## Quick Start

```bash
# Test with 3 cases first
python evaluate/quick_eval.py --limit 3

# Run full evaluation (~3-4 hours)
python evaluate/quick_eval.py

# Test specific case (e.g., case-162)
python evaluate/quick_eval.py --limit 1 --skip 161
```

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

