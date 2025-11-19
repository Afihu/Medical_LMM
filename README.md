# NO RAG - Medical AI Evaluation System (Local Setup)

## Prerequisites

- Python 3.11 (Highly recommended to avoid known gRPC crashes on Windows with Python 3.13).

- A Google Cloud API Key with access to Gemini/Gemma.

## Setup Instructions

### 1. Create a Virtual Environment

Open your terminal (PowerShell or Command Prompt) in this folder and run:

```powershell
# Create the environment named .venv
py -3.11 -m venv .venv

# Activate the environment
# Windows (PowerShell):
.\.venv\Scripts\Activate
# Mac/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

With the environment activated, install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a file named .env in this folder and add your keys:

```text
GOOGLE_API_KEY=your_actual_api_key_here
DIAGNOSIS_MODEL=gemini-2.5-flash
```

## Usage

Run the script using python quick_eval.py. The system uses a "Burst Mode" that runs as fast as possible and automatically pauses if it hits the Free Tier rate limit.

### Common Commands

Run a small test (first 2 cases):

```bash
python quick_eval.py --limit 2
```

Run specific cases (e.g., cases 30 to 40):

```bash
python quick_eval.py --limit 10 --skip 30
```

Run the full evaluation:

```bash
python quick_eval.py
```

## Output

For every run, a new folder is created (e.g., session_20251112_093000) containing:

- `results.csv`: Summary table (Case ID, Relevancy, Correctness).

- `detailed_results.json`: Full data including prompts, AI reasoning, and raw scores.

- `summary_stats.txt`: Final average scores for the batch.
 