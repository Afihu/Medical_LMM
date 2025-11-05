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

def flatten_answer(answer_list):
    # If answer_list is a list of dicts, flatten to a single string
    if isinstance(answer_list, list):
        parts = []
        for entry in answer_list:
            disease = entry.get("disease", "")
            reasoning = entry.get("reasoning", "")
            likelihood = entry.get("likelihood", "")
            # Only include non-empty fields
            part = f"Disease: {disease}; Reasoning: {reasoning}; Likelihood: {likelihood}"
            parts.append(part.strip())
        return "\n".join(parts)
    # If answer_list is already a string, just return it
    return str(answer_list)

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
    ai_answer = diag.get('ai_response', {}).get('differential_diagnosis', [])
    retrieved_cases = diag.get('retrieved_cases', {})
    # Rename context to retrieved_contexts
    retrieved_contexts = [case['text'] for case in retrieved_cases.values()] if retrieved_cases else []
    ground_truth = prompt_to_gt.get(key, None)
    # Flatten answer
    answer_text = flatten_answer(ai_answer)
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