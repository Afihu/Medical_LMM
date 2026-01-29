# Medical_LMM

## Getting started
- The current working pipeline is the embedding one. It's end-to-end from data extraction from PDFs to embedding generation and storage in Qdrant.
- Please follow the instructions below to set up the environment and run the project.
### Environment Setup
- We will be using a virtual environment to manage our dependencies. You can use either `uv` or `pip` as your package manager. These instructions are for `uv`. For `pip`, please refer to the "With `pip`" section below.

```bash
# Sync your dependencies
uv sync

# Add the `.env` file to the root directory

# Create `/source_materials` directory in the root and put in the PDF files you want to process.
mkdir source_materials

# Run pdf extraction module
uv run "scripts/data_extraction_module/pdf_extraction_orchestrator.py"

# Run embedding generation module
uv run "scripts/embedding_generation_module/orchestrators/embedding_orchestrator.py"

# NOTE: The first time you run this script, the models will be downloaded and cached, which may take some time.

# After running these, you should see the extracted data in `extracted_data/` and the generated embeddings in `staged_embeddings/`.

# Run upload service to Qdrant
uv run "scripts/qdrant_services/upload.py"
```

### Packages

- **This project requires Python 3.13 or higher (as in `.python-version`). Please install it and set up the paths before continuing.**

- The required packages are in `pyproject.toml` for `uv` and `requirements.txt` for `pip`.

### With `uv`
- I recommend using `uv` as our package manager. Please install it as follows:
```bash
pip install uv

# After having `uv` installed, you can install the required dependencies by running:
uv sync 

# This will install all the packages listed in `pyproject.toml` and create a `.env` file for you.
```

## With `pip`
- If you prefer to use `pip`, you can install the required dependencies by following these steps:

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv

# 2. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install the required dependencies
pip install -r requirements.txt
```

### Once you have the packages installed
- With `uv`, can run the project directly using:
```bash
uv run "path/to/your/script.py"
# This will automatically activate the virtual environment and run the script you want.
```

- With normal python, you can run the project using:
```bash
# Make sure your virtual environment is activated if you created one (as instructed above).
python path/to/your/script.py
```