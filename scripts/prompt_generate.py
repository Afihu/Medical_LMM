"""
prompt_generate.py
------------------
Combines the base prompt (from prompt.txt) with the top 5 retrieved cases in /cases/.
This script is used by main.py before sending the prompt to Gemini.
"""

import os
import json

# --- Path configuration ---
# Get the root directory of the project (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # as this will return /scripts, so we wrap in another one
PROMPT_FILE = os.path.join(PROJECT_ROOT, "scripts/prompt.txt")
CASES_FOLDER = os.path.join(PROJECT_ROOT, "cases")
# -------------------------------------------------------------------

def load_cases():
    """Load all case JSON files from /cases folder."""
    cases = []
    if not os.path.exists(CASES_FOLDER):
        print(f"Warning: '{CASES_FOLDER}' directory not found.")
        return cases

    for filename in sorted(os.listdir(CASES_FOLDER)):
        if filename.endswith(".json"):
            path = os.path.join(CASES_FOLDER, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cases.append(data)
            except Exception as e:
                print(f"Could not read {filename}: {e}")
    return cases


def format_cases(cases):
    """Format all cases as markdown text."""
    if not cases:
        return "No similar cases were provided for context."

    formatted = []
    for i, case in enumerate(cases):
        payload = case.get("payload", {})
        case_md = (
            f"\n**Case {i+1} (ID: {case.get('id', 'N/A')})**\n"
            f"- **Diagnosis:** {payload.get('diagnosis', 'N/A')}\n"
            f"- **Symptoms:** {payload.get('symptoms', 'N/A')}\n"
            f"- **History:** {payload.get('history', 'N/A')}\n"
        )
        formatted.append(case_md)
    return "\n".join(formatted)


def generate_prompt(user_input):
    """Generate the final prompt string by merging base prompt + cases."""
    # Load base prompt
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"Missing {PROMPT_FILE}")

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    # Load cases
    cases = load_cases()
    cases_section = format_cases(cases)

    # Replace placeholders
    final_prompt = (
        base_prompt
        .replace("{user_input}", user_input.strip())
        .replace("{cases_section}", cases_section)
    )

    return final_prompt


# if __name__ == "__main__":
#     # Example run
#     example_input = "Patient reports shortness of breath and mild chest discomfort."
#     prompt = generate_prompt(example_input)
#     print("\nGenerated Prompt:\n")
#     print(prompt)
