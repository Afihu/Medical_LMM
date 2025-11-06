# Batch Evaluation Scripts

Automated RAGAS evaluation for Medical_LMM test cases.

## Quick Start

```bash
# Test with 3 cases (recommended first run)
python evaluate/quick_eval.py --limit 3

# Evaluate all 22 test cases (~20-30 minutes)
python evaluate/quick_eval.py

# Use existing diagnosis files (skip diagnosis step)
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

## RAGAS Metrics

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Context Precision** | Retrieved contexts relevant? | > 0.8 |
| **Context Recall** | All needed info retrieved? | > 0.8 |
| **Faithfulness** | Answer grounded in context? | > 0.7 |
| **Answer Relevancy** | Answer addresses question? | > 0.8 |

## Requirements

```bash
pip install ragas langchain-google-genai langchain-huggingface qdrant-client sentence-transformers
```

Ensure `.env` contains: `GEMINI_API_KEY=your_api_key_here`

## Troubleshooting

**Missing API Key**: Add `GEMINI_API_KEY` to `.env` file  
**Rate Limiting**: Use `--limit` to run smaller batches  
**Module Errors**: Install all requirements above

## Performance

- ~30-60 seconds per case (6-10 Gemini API calls each)
- 22 test cases = ~20-30 minutes total
- Safe to interrupt (progress saved after each case)
