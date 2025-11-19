"""
prompt_generate.py (Refactored)
---------------------------------------------------------
Combines the base prompt (from prompt.txt) with retrieved cases
from /temp_query_data/<session_id>/retrieved_cases/.

Used by main_streamlit.py before sending the final prompt to Gemini.
"""

import os
import json
import base64
from io import BytesIO
from PIL import Image


# --- Path configuration ---
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PROMPT_FILE = os.path.join(PROJECT_ROOT, "/main_runtime/prompt.txt")
# TEMP_QUERY_DATA = os.path.join(PROJECT_ROOT, "temp_query_data")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPT_DIR = os.path.join(PROJECT_ROOT, "scripts", "main_runtime")
PROMPT_FILE = os.path.join(PROMPT_DIR, "prompt.txt")  # Default for main pipeline
TEMP_QUERY_DATA = os.path.join(PROJECT_ROOT, "temp_query_data")

# Evaluation-specific prompt files
PROMPT_FILES = {
    None: os.path.join(PROMPT_DIR, "prompt.txt"),           # Default (main pipeline)
    "internal": os.path.join(PROMPT_DIR, "prompt_internal.txt"),
    "rag": os.path.join(PROMPT_DIR, "prompt_rag.txt"),
    "hybrid": os.path.join(PROMPT_DIR, "prompt_hybrid.txt"),
}


# -------------------------------------------------------------------

def load_retrieved_cases(retrieved_cases_dir):
    """Load all case JSON files from the retrieved_cases folder."""
    cases = []
    if not os.path.exists(retrieved_cases_dir):
        print(f"[WARN] Retrieved cases directory not found: {retrieved_cases_dir}")
        return cases

    for filename in sorted(os.listdir(retrieved_cases_dir)):
        if filename.endswith(".json"):
            path = os.path.join(retrieved_cases_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cases.append(data)
            except Exception as e:
                print(f"[WARN] Could not load {filename}: {e}")
    return cases


def format_cases_markdown(cases):
    """Format merged (text + image) cases into Markdown for Gemini prompt."""
    if not cases:
        return "No similar cases were found for context."

    formatted_sections = ["### Retrieved Similar Cases:\n"]
    for i, case in enumerate(cases, start=1):
        cid = case.get("case_id", "Unknown")
        formatted_sections.append(f"#### Case {i} (ID: {cid})")

        # --- Text section ---
        text_payload = case.get("text", {})
        if text_payload:
            formatted_sections.append("**Text Summary:**")
            try:
                text_json = json.dumps(text_payload, indent=4, ensure_ascii=False)
                formatted_sections.append(f"```json\n{text_json}\n```")
            except Exception as e:
                formatted_sections.append(f"Error formatting text payload: {e}")
        else:
            formatted_sections.append("_No text summary available._")

        # --- Image section ---
        images = case.get("images", [])
        if images:
            formatted_sections.append("**Associated Images:**")
            for j, img_payload in enumerate(images, start=1):
                try:
                    caption = img_payload.get("Caption", "(no caption)")
                    formatted_sections.append(f"- Image {j}: {caption}")
                except Exception as e:
                    formatted_sections.append(f"- Error reading image payload: {e}")
        else:
            formatted_sections.append("_No associated images._")

        formatted_sections.append("\n")

    return "\n".join(formatted_sections)

def decode_case_images(cases, session_id):
    """
    Decode base64-encoded images from retrieved cases and save them
    under /temp_query_data/<session_id>/decoded_images/.
    Returns a list of decoded image file paths.
    """
    decoded_paths = []
    output_dir = os.path.join(TEMP_QUERY_DATA, session_id, "decoded_images")
    os.makedirs(output_dir, exist_ok=True)

    for case in cases:
        cid = case.get("case_id", "unknown")
        for idx, img_payload in enumerate(case.get("images", []), start=1):
            if "image_base64" in img_payload:
                try:
                    data = base64.b64decode(img_payload["image_base64"])
                    img = Image.open(BytesIO(data))
                    filename = f"{cid}_img_{idx}.png"
                    out_path = os.path.join(output_dir, filename)
                    img.save(out_path)
                    decoded_paths.append(out_path)
                except Exception as e:
                    print(f"[WARN] Could not decode image for case {cid}: {e}")
    return decoded_paths

def generate_prompt(user_input, session_id="default_session", eval_mode=None):
    """
    Generate the final structured prompt for LLM.
    
    Args:
        user_input (str): User's text query
        session_id (str): The ID of the current query session
        eval_mode (str, optional): Evaluation mode ("internal", "rag", "hybrid", or None for default)
                                   - None: Uses prompt.txt (default, main pipeline)
                                   - "internal": Uses prompt_internal.txt (no context needed)
                                   - "rag": Uses prompt_rag.txt (context only)
                                   - "hybrid": Uses prompt_hybrid.txt (context + internal knowledge)

    Returns:
        tuple[str, list[str]]: (final_prompt_text, decoded_image_paths)
    """
    # Select prompt template based on eval_mode
    prompt_file = PROMPT_FILES.get(eval_mode, PROMPT_FILE)
    
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Missing {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    # Load retrieved cases from current session (only for modes that use context)
    cases_section = ""
    if eval_mode != "internal":
        retrieved_dir = os.path.join(TEMP_QUERY_DATA, session_id, "retrieved_cases")
        cases = load_retrieved_cases(retrieved_dir)
        cases_section = format_cases_markdown(cases)
    else:
        # For internal mode, provide empty cases section (won't be in template, but safe)
        cases_section = ""

    # Replace placeholders
    final_prompt = (
        base_prompt
        .replace("{user_input}", user_input.strip())
        .replace("{cases_section}", cases_section)
    )

    # Decode and collect image paths (only for modes that use images)
    decoded_images_path = []
    if eval_mode != "internal":
        retrieved_dir = os.path.join(TEMP_QUERY_DATA, session_id, "retrieved_cases")
        cases = load_retrieved_cases(retrieved_dir)
        decoded_images_path = decode_case_images(cases, session_id)
    
    return final_prompt, decoded_images_path

# --- Test run (optional) ---
if __name__ == "__main__":
    example_input = "Patient reports blurred vision and fatigue."
    prompt = generate_prompt(example_input)
    print("\nGenerated Prompt:\n")
    print(prompt)
