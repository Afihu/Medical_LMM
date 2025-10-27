"""
main.py
-------
Main driver script for the Medical_LMM system.

Flow:
1. Receive new patient text input.
2. Use CLIP (later) to generate embedding vector.
3. Query Qdrant Cloud for top-5 similar cases (via scripts/query.py).
4. Save them to /cases folder as JSON.
5. Construct the prompt (via scripts/prompt_generate.py).
6. Send the prompt to Gemini API for diagnostic analysis.
7. Save Gemini response to /responses as Markdown.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Custom modules
from scripts.query import run_query
from scripts.prompt_generate import generate_prompt

# --- Path configuration ---
# Get the project root (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CASES_FOLDER = os.path.join(PROJECT_ROOT, "cases")
RESPONSES_FOLDER = os.path.join(PROJECT_ROOT, "responses")

# --- Configuration ---
MODEL_NAME = "models/gemini-2.5-pro"

# --- Setup ---
def setup():
    """Configure Gemini API and ensure directories exist."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")

    genai.configure(api_key=api_key)

    os.makedirs(RESPONSES_FOLDER, exist_ok=True)
    os.makedirs(CASES_FOLDER, exist_ok=True)

    print("Setup complete. Gemini and directories ready.\n")


# --- Main Loop ---
def main():
    setup()
    model = genai.GenerativeModel(MODEL_NAME)
    print("--- Medical LMM Analysis ---")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        user_input = input("Enter new patient's symptoms and history: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting program. Goodbye!")
            break

        if not user_input:
            print("Please enter a valid description.")
            continue

        # Step 1: (Placeholder) Generate embedding for query
        # Replace this with real encoder later
        query_vector = [0.12, -0.45, 0.78, 0.66]
        run_query(query_vector)

        # Step 2: Generate the full prompt for Gemini
        print("Building prompt from retrieved cases...")
        final_prompt = generate_prompt(user_input)

        # Step 3: Send to Gemini API
        try:
            print("Generating Gemini analysis... (this may take a moment)")
            response = model.generate_content(final_prompt)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.md"
            filepath = os.path.join(RESPONSES_FOLDER, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)

            print(f"\nAnalysis complete. Report saved to: {filepath}\n")

        except Exception as e:
            print(f"Error during Gemini generation: {e}")


# --- Entry Point ---
if __name__ == "__main__":
    main()