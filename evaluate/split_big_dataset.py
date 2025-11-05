import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "ragas_dataset.json"
OUTPUT_DIR = Path(__file__).parent / "split_cases"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    cases = json.load(f)

for idx, case in enumerate(cases, 1):
    out_path = OUTPUT_DIR / f"case_{idx}.json"
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(case, out_f, indent=2, ensure_ascii=False)

print(f"Split {len(cases)} cases into {OUTPUT_DIR}")