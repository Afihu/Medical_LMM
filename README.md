# Medical_LMM

## Getting started

### Packages

- **This project requires Python 3.13 or higher (as in `.python-version`). Please install it and set up the paths before continuing.**

- The required packages are:
    - `google-generativeai`
    - `open-clip-pytorch`
    - `Pillow`
    - `python-dotenv`
    - `torch`
    - `qdrant-client`

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
uv run main.py
# This will automatically activate the virtual environment and run the `main.py` script.
```

- With normal python, you can run the project using:
```bash
# Make sure your virtual environment is activated if you created one (as instructed above).
python main.py
```

# Note:
- Please keep the `main.py` file in the root directory as the entry point of the project.
- New packages should be added with `uv add <package-name>` if you are using `uv`, or by updating the `requirements.txt` file if you are using `pip`.

## MCP for AI Coding Assistant
- If you are using AI coding assistants like GitHub Copilot, I suggest the following MCP server to keep the agent up-to-date with the dependencies' documentation:
    - Main site: https://github.com/mcp?utm_source=vscode-website&utm_campaign=mcp-registry-server-launch-2025
        - DeepWiki - For retrieving the documentation of the packages or tools.
        - Other MCP agents such as `context7` or `serena` can also be useful, but, please use the officical MCP registry to find the most suitable ones.
    - Sample prompt:
    ```
    Please use `deepwiki` to look up the documentation regarding the usage of `Multi-Vector` in Qdrant.
    ``` 
