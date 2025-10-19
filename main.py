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

        # Step 1: (Placeholder) Generate CLIP embedding for query
        # Replace this with real CLIP encoder later
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


# import google.generativeai as genai
# import os
# import json
# from datetime import datetime
# from dotenv import load_dotenv

# # --- LOAD YOUR API KEY HERE ---
# # You can also set it as an environment variable named "GEMINI_API_KEY"
# # Execute the command to load the file
# load_dotenv()

# # --- Configuration ---
# API_KEY = os.getenv("GEMINI_API_KEY")
# CASES_FOLDER = "cases"
# RESPONSES_FOLDER = "responses"
# MODEL_NAME = 'models/gemini-2.5-pro'

# # --- Setup ---
# def setup():
#     """Configures the API and creates necessary folders."""
#     try:
#         genai.configure(api_key=API_KEY)    # Use the API key
#     except Exception as e:
#         print(f"Error configuring the API: {e}")
#         exit()

#     # Create the responses folder if it doesn't exist
#     if not os.path.exists(RESPONSES_FOLDER):
#         os.makedirs(RESPONSES_FOLDER)
#         print(f"Created directory: '{RESPONSES_FOLDER}'")

# def load_similar_cases():
#     """Loads all JSON case files from the /cases directory."""
#     cases = []
#     if not os.path.exists(CASES_FOLDER):
#         print(f"Warning: '{CASES_FOLDER}' directory not found. Continuing without similar cases.")
#         return cases

#     print(f"Loading similar cases from '{CASES_FOLDER}'...")
#     for filename in os.listdir(CASES_FOLDER):
#         if filename.endswith('.json'):
#             filepath = os.path.join(CASES_FOLDER, filename)
#             try:
#                 with open(filepath, 'r') as f:
#                     data = json.load(f)
#                     cases.append(data)
#             except Exception as e:
#                 print(f"Warning: Could not read or parse {filename}. Error: {e}")
    
#     print(f"Found {len(cases)} similar cases.")
#     return cases

# # --- Main Program Loop ---
# def main():
#     """
#     Main function to run the medical analysis chat loop.
#     """
#     setup()
#     model = genai.GenerativeModel(MODEL_NAME)
    
#     print("\n--- Medical LMM Analysis with Context ---")
#     print("Provide a textual description of the patient's symptoms.")
#     print("The system will use similar cases from the '/cases' folder for context.")
#     print("Type 'quit' or 'exit' to end.")

#     while True:
#         # Get input from the user for the new case
#         new_case_description = input("\nEnter new patient's symptoms and history: ")

#         if new_case_description.lower() in ['quit', 'exit']:
#             print("Exiting the program. Goodbye!")
#             break

#         if not new_case_description:
#             print("Please enter a description.")
#             continue

#         # Load cases every time to catch any updates
#         similar_cases = load_similar_cases()

#         # --- Prompt Engineering --- (example, because we haven't figured it out yet)
#         prompt_parts = [
#             "You are a world-class medical diagnostician AI.",
#             "Analyze the 'New Patient Information' in the context of the 'Similar Past Cases' provided.",
#             "Your task is to provide a detailed differential diagnosis and a recommended action plan.",
#             "Format your entire response as a professional medical report in Markdown.\n",
#             "---",
#             "### New Patient Information:",
#             f"- **Symptoms & History:** {new_case_description}\n",
#             "---",
#             "### Similar Past Cases for Context:",
#         ]

#         if similar_cases:
#             for i, case in enumerate(similar_cases):
#                 payload = case.get('payload', {})
#                 case_info = (
#                     f"\n**Case {i+1} (ID: {case.get('id', 'N/A')})**\n"
#                     f"- **Diagnosis:** {payload.get('diagnosis', 'N/A')}\n"
#                     f"- **Symptoms:** {payload.get('symptoms', 'N/A')}\n"
#                     f"- **History:** {payload.get('history', 'N/A')}\n"
#                 )
#                 prompt_parts.append(case_info)
#         else:
#             prompt_parts.append("No similar cases were provided for context.")
        
#         prompt_parts.append("---\n")
#         prompt_parts.append("### Analysis and Report:\n")
#         prompt_parts.append("Based on all the information, provide your detailed analysis below.\n")

#         final_prompt = "\n".join(prompt_parts)
        
#         # --- Generate and Save Response ---
#         try:
#             print("\nGenerating analysis... (This may take a moment)")
#             response = model.generate_content(final_prompt)
            
#             # Create a unique filename for the markdown file
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"analysis_{timestamp}.md"
#             filepath = os.path.join(RESPONSES_FOLDER, filename)

#             # Save the response to the markdown file
#             with open(filepath, 'w', encoding='utf-8') as f:
#                 f.write(response.text)
            
#             print(f"\nAnalysis complete. Report saved to: '{filepath}'")

#         except Exception as e:
#             print(f"\nAn error occurred while generating the response: {e}")

# # Run the main function
# if __name__ == '__main__':
#     main()