"""
prompt_generate.py
------------------
Combines the base prompt (from prompt.txt) with retrieved cases
from both /cases-text/ and /cases-image/ folders.
Used by main_streamlit.py (or main.py) before sending the final
prompt to Gemini for analysis.
"""

import os
import json
import base64

# --- Path configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_FILE = os.path.join(PROJECT_ROOT, "scripts/prompt.txt")
CASES_FOLDER_TEXT = os.path.join(PROJECT_ROOT, "cases_text")
CASES_FOLDER_IMAGE = os.path.join(PROJECT_ROOT, "cases_image")
DECODED_IMG_FOLDER = os.path.join(PROJECT_ROOT, "decoded_images")

# --- Ensure folder exists ---
os.makedirs(DECODED_IMG_FOLDER, exist_ok=True)

# -------------------------------------------------------------------

def load_cases(folder_path):
    """Load all case JSON files from the specified folder."""
    cases = []
    if not os.path.exists(folder_path):
        print(f"Warning: '{folder_path}' not found.")
        return cases

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            path = os.path.join(folder_path, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cases.append(data)
            except Exception as e:
                print(f"Could not read {filename}: {e}")
    return cases

def decode_and_save_image(base64_str, case_id, index):
    """Decode base64 image and save to /decoded_images. Returns the saved path."""
    try:
        image_bytes = base64.b64decode(base64_str)
        image_path = os.path.join(DECODED_IMG_FOLDER, f"{case_id}_{index}.png")
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        return image_path
    except Exception as e:
        print(f"Error decoding image for {case_id}: {e}")
        return None

def format_cases_markdown(cases_text, cases_image):
    """Format both text-based and image-based cases into Markdown sections."""
    if not cases_text and not cases_image:
        return "No similar cases were found for context."

    formatted_sections = []
    decoded_image_paths = []

    # --- Text-based cases ---
    if cases_text:
        formatted_sections.append("#### Text-Based Similar Cases:")
        for i, case in enumerate(cases_text):
            try:
                case_json = json.dumps(case, indent=4, ensure_ascii=False)
                formatted_sections.append(f"**Text Case {i+1}:**\n```json\n{case_json}\n```")
            except Exception as e:
                formatted_sections.append(f"Error formatting case {i+1}: {e}")

    # --- Image-based cases ---
    if cases_image:
        formatted_sections.append("\n#### Image-Based Similar Cases:")
        for i, case in enumerate(cases_image):
            try:
                case_id = str(case.get("id", f"img_{i+1}"))
                payload = case.get("payload", {})
                image_b64 = payload.get("image")

                case_md = [f"**Image Case {i+1} (ID: {case_id})**"]

                # Decode and attach image if present
                if image_b64:
                    image_path = decode_and_save_image(image_b64, case_id, i + 1)
                    if image_path:
                        # Collect decoded path for later attachment
                        decoded_image_paths.append(image_path)
                        # show relative path in markdown for human readability (optional)
                        rel_path = os.path.relpath(image_path, PROJECT_ROOT)
                        case_md.append(f"![Case Image]({rel_path})")

                # Include full metadata for reference
                case_json = json.dumps(case, indent=4, ensure_ascii=False)
                case_md.append(f"```json\n{case_json}\n```")

                formatted_sections.append("\n".join(case_md))
            except Exception as e:
                formatted_sections.append(f"Error formatting image case {i+1}: {e}")

    return "\n\n".join(formatted_sections), decoded_image_paths


def generate_prompt(user_input):
    """
    Generate the final prompt string by merging base prompt + text & image cases.
    Returns: (final_prompt_text, list_of_decoded_image_paths)
    """
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"Missing {PROMPT_FILE}")

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    # Load cases from both sources
    text_cases = load_cases(CASES_FOLDER_TEXT)
    image_cases = load_cases(CASES_FOLDER_IMAGE)

    # Combine into one section and decode images (collect decoded paths)
    cases_section, decoded_images_path = format_cases_markdown(text_cases, image_cases)

    # Replace placeholders in the base prompt
    final_prompt = (
        base_prompt
        .replace("{user_input}", (user_input or "").strip())
        .replace("{cases_section}", cases_section)
    )

    return final_prompt, decoded_images_path


# --- Test run (optional) ---
if __name__ == "__main__":
    example_input = "Patient reports blurred vision and fatigue."
    prompt = generate_prompt(example_input)
    print("\nGenerated Prompt:\n")
    print(prompt[:1000])    # print first 1000 chars to avoid flooding
