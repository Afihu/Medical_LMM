import json
import re
from pathlib import Path

def normalize(text):
    # Remove escape characters, quotes, and lowercase
    text = re.sub(r'[\\"]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def first_n_words(text, n=12):
    words = normalize(text).split()
    return ' '.join(words[:n])

def flatten_answer(ai_response):
    """
    Flatten the entire AI response (including analysis and differential diagnosis) 
    into a single coherent string for RAGAS evaluation.
    """
    if isinstance(ai_response, str):
        return ai_response
    
    if not isinstance(ai_response, dict):
        return str(ai_response)
    
    output_parts = []
    
    # Extract analysis steps if available
    analysis = ai_response.get("analysis", [])
    if analysis and isinstance(analysis, list):
        output_parts.append("CLINICAL ANALYSIS:")
        for step in analysis:
            if isinstance(step, dict):
                step_name = step.get("step", "")
                content = step.get("content", "")
                if step_name and content:
                    output_parts.append(f"\n{step_name}:")
                    output_parts.append(content)
    
    # Extract differential diagnosis
    diff_diagnosis = ai_response.get("differential_diagnosis", [])
    if diff_diagnosis and isinstance(diff_diagnosis, list):
        output_parts.append("\n\nDIFFERENTIAL DIAGNOSIS:")
        for idx, entry in enumerate(diff_diagnosis, 1):
            if isinstance(entry, dict):
                disease = entry.get("disease", "Unknown")
                reasoning = entry.get("reasoning", "")
                likelihood = entry.get("likelihood", "")
                
                output_parts.append(f"\n{idx}. {disease} (Likelihood: {likelihood})")
                if reasoning:
                    output_parts.append(f"   Reasoning: {reasoning}")
    
    # If no structured data found, try to extract any text content
    if not output_parts:
        return str(ai_response)
    
    return "\n".join(output_parts)

BASE_DIR = Path(__file__).parent.parent
CASES_PATH = BASE_DIR / "test_cases" / "cases.json"
DIAGNOSED_CASES_DIR = BASE_DIR / "diagnosed_cases"

with open(CASES_PATH, 'r', encoding='utf-8') as f:
    cases = json.load(f)

# Build mapping from normalized first N words to diagnosis
N = 12
prompt_to_gt = {first_n_words(case['prompt'], N): case['diagnosis'] for case in cases if 'prompt' in case}

diagnosis_files = DIAGNOSED_CASES_DIR.glob("diagnosis_*.json")

ragas_dataset = []
skipped_files = []
for file in diagnosis_files:
    with open(file, 'r', encoding='utf-8') as f:
        diag = json.load(f)
    prompt = diag.get('user_query', {}).get('text', '')
    key = first_n_words(prompt, N)
    ai_response = diag.get('ai_response', {})
    retrieved_cases = diag.get('retrieved_cases', {})
    # Rename context to retrieved_contexts
    retrieved_contexts = [case['text'] for case in retrieved_cases.values()] if retrieved_cases else []
    ground_truth = prompt_to_gt.get(key, None)
    # Flatten the entire AI response into a string
    answer_text = flatten_answer(ai_response)
    if ground_truth:
        ragas_dataset.append({
            "question": prompt,
            "retrieved_contexts": retrieved_contexts,
            "answer": answer_text,
            "ground_truth": ground_truth
        })
    else:
        print(f"\nSkipped: {file.name}")
        print("Diagnosis prompt first N normalized words:")
        print(repr(key))
        print("\nAvailable normalized keys in cases.json:")
        for k in prompt_to_gt.keys():
            print(repr(k))
        skipped_files.append(file.name)

with open(BASE_DIR / "evaluate" / "ragas_dataset.json", "w", encoding="utf-8") as f:
    json.dump(ragas_dataset, f, indent=2, ensure_ascii=False)
print(f"Saved {len(ragas_dataset)} entries to ragas_dataset.json")
print(f"Skipped {len(skipped_files)} files: {skipped_files}")