# Batch Evaluation Scripts

Automated RAGAS evaluation for Medical_LMM test cases.

## Quick Start

```bash
# Test with 3 cases (recommended first run)
python evaluate/quick_eval.py --limit 3

# Evaluate all 22 test cases (~45-90 minutes)
python evaluate/quick_eval.py

# Test specific case (e.g., case-022, which is index 21)
python evaluate/quick_eval.py --limit 1 --skip 21

# Skip diagnosis step, use existing diagnosis files
python evaluate/quick_eval.py --skip-diagnosis
```

## What It Does

Automates the entire pipeline:
1. Runs diagnosis on each test case (Qdrant + Gemini)
2. Matches with ground truth from `test_cases/cases.json`
3. Evaluates with RAGAS metrics (context precision/recall, faithfulness, answer relevancy)
4. Generates reports in 3 formats: JSON, CSV, Markdown

## Output Files

Results saved in `evaluate/batch_results/`:
- **CSV** - Easy analysis in Excel/Google Sheets (9 columns: case_id, diagnosis, status, 4 metrics, timestamp, error)
- **JSON** - Full metadata with scores and contexts
- **Markdown** - Human-readable summary report

## Command Options

- `--limit N` - Evaluate only first N cases (useful for testing)
- `--skip N` - Skip first N cases (useful for testing specific cases)
- `--skip-diagnosis` - Skip diagnosis generation, use existing files from `diagnosed_cases/`

**Examples:**
```bash
# Test first 5 cases
python evaluate/quick_eval.py --limit 5

# Test case-010 only (skip first 9, limit to 1)
python evaluate/quick_eval.py --skip 9 --limit 1

# Re-evaluate using existing diagnosis files
python evaluate/quick_eval.py --skip-diagnosis
```



## Requirements

```bash
pip install ragas langchain-google-genai langchain-huggingface qdrant-client sentence-transformers
```

Ensure `.env` contains: `GEMINI_API_KEY=your_api_key_here`

## Troubleshooting

**Missing API Key**: Add `GEMINI_API_KEY` to `.env` file  
**Rate Limiting**: Use `--limit` to run smaller batches  
**Module Errors**: Install all requirements above

