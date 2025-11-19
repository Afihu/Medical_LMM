import os
import json
import time
import argparse
import sys
import asyncio
import warnings
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 0. WINDOWS STABILITY PATCHES
# ==========================================
import nest_asyncio
nest_asyncio.apply()
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
import google.generativeai as genai
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
DIAGNOSIS_MODEL_NAME = os.getenv("DIAGNOSIS_MODEL", "gemini-1.5-flash")
EVALUATION_MODEL_NAME = "gemini-2.5-flash"

if not API_KEY:
    print("[ERROR] GOOGLE_API_KEY missing in .env")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_file(name):
    path = BASE_DIR / name
    if not path.exists():
        print(f"[ERROR] Missing file: {name}")
        sys.exit(1)
    if name.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    return path.read_text(encoding='utf-8').strip()

def safe_evaluate(dataset, metrics, wrappers, retries=5):
    """
    Optimized: Runs fast, but catches 429 errors and sleeps only when necessary.
    """
    llm, embed = wrappers
    # NOTE: If you have a PAID API key, change max_workers to 4 for faster speed.
    # For Free Tier, keep max_workers=1 to avoid instant crashes.
    config = RunConfig(max_workers=1, timeout=180)
    
    for attempt in range(retries):
        try:
            return evaluate(
                dataset, 
                metrics=metrics, 
                llm=llm, 
                embeddings=embed,
                run_config=config,
                raise_exceptions=True
            )
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                # Smart Wait: If we hit the limit, wait enough to reset the minute window
                wait_time = 65 
                print(f"      [WARN] Rate limit hit (Burst complete). Pausing {wait_time}s to reset quota...")
                time.sleep(wait_time)
            elif "AttributeError" in error_msg:
                 # Ignore Windows/gRPC cleanup bugs
                 return None
            else:
                print(f"      [ERROR] {e}")
                if attempt == retries - 1: return None
                time.sleep(2)
    return None

def get_score(result, metric):
    if result is None: return 0.0
    try:
        val = result[metric]
    except (KeyError, TypeError, AttributeError):
        return 0.0
    return val[0] if isinstance(val, list) else val

# ==========================================
# 3. SETUP MODELS
# ==========================================
print(f"[INFO] Diagnosis Model: {DIAGNOSIS_MODEL_NAME}")
print(f"[INFO] Ragas Evaluator: {EVALUATION_MODEL_NAME}")

doctor = genai.GenerativeModel(DIAGNOSIS_MODEL_NAME, system_instruction=load_file('prompt.txt'))

ragas_llm_raw = ChatGoogleGenerativeAI(model=EVALUATION_MODEL_NAME, temperature=0.0)
ragas_embed_raw = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

ragas_llm = LangchainLLMWrapper(ragas_llm_raw)
ragas_embed = LangchainEmbeddingsWrapper(ragas_embed_raw)

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = BASE_DIR / f"session_{timestamp_str}"
    os.makedirs(session_dir, exist_ok=True)
    print(f"[INFO] Session: {session_dir.name}")

    csv_path = session_dir / "results.csv"
    json_path = session_dir / "detailed_results.json"
    summary_path = session_dir / "summary_stats.txt"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Case ID", "Timestamp", "Answer Relevancy", "Answer Correctness"])

    data = load_file('augmented_test.json')
    start = args.skip
    end = start + args.limit if args.limit else len(data)
    subset = data[start:end]

    print(f"[INFO] Processing {len(subset)} cases (Burst Mode)...")
    results_list = []

    for i, case in enumerate(subset):
        case_id = case.get('id', 'unknown')
        current_time = datetime.now().isoformat()
        print(f"\n[{i+1}/{len(subset)}] Case: {case_id}")

        # A. Generate
        try:
            # Tiny sleep to prevent hitting limits on generation step
            time.sleep(1) 
            response = doctor.generate_content(case['prompt'])
            full_text = response.text.strip()
        except Exception as e:
            print(f"   [ERROR] Generation Failed: {e}")
            # If generation fails due to limit, we must wait
            if "429" in str(e):
                print("   [WARN] Gen Limit Hit. Sleeping 60s...")
                time.sleep(60)
            continue

        # B. Parse
        try:
            clean_json = full_text.replace('```json', '').replace('```', '').strip()
            parsed = json.loads(clean_json)
            # Split by " - " (space-hyphen-space) to preserve names like "Tick-Bite"
            clean_name = parsed.get('concise_answer', '').split('(')[0].split(' - ')[0].strip()
            concise = parsed.get('concise_answer', '')
            reasoning_text = "\n".join([f"- {item.get('disease')}: {item.get('reasoning')}" for item in parsed.get('differential_diagnosis', [])])
            narrative = f"Answer: {concise}\nReasoning:\n{reasoning_text}"
            ai_json_response = parsed
        except:
            clean_name = full_text[:50]
            narrative = full_text
            ai_json_response = {"raw_text": full_text}

        # C. Relevancy (Removed fixed sleep)
        print("   [INFO] Running Relevancy...")
        res_rel = safe_evaluate(
            Dataset.from_dict({'question': [case['prompt']], 'contexts': [[]], 'answer': [narrative]}),
            [answer_relevancy], (ragas_llm, ragas_embed)
        )
        
        # D. Correctness (Removed fixed sleep)
        print("   [INFO] Running Correctness...")
        res_corr = safe_evaluate(
            Dataset.from_dict({
                'question': [case['prompt']], 'ground_truths': [[case['diagnosis']]], 
                'contexts': [[]], 'answer': [clean_name], 'reference': [case['diagnosis']]
            }),
            [answer_correctness], (ragas_llm, ragas_embed)
        )

        s_rel = get_score(res_rel, 'answer_relevancy')
        s_corr = get_score(res_corr, 'answer_correctness')
        
        print(f"   [RESULT] Rel: {s_rel:.3f} | Corr: {s_corr:.3f}")

        # Save
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([case_id, current_time, f"{s_rel:.4f}", f"{s_corr:.4f}"])

        record = {
            "case_id": case_id, "timestamp": current_time,
            "prompt": case['prompt'], "ai_response_json": ai_json_response,
            "ground_truth": case['diagnosis'],
            "scores": {"answer_relevancy": s_rel, "answer_correctness": s_corr}
        }
        results_list.append(record)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)

    # Summary
    if results_list:
        avg_rel = sum(r['scores']['answer_relevancy'] for r in results_list) / len(results_list)
        avg_corr = sum(r['scores']['answer_correctness'] for r in results_list) / len(results_list)
        summary_text = f"Total Cases: {len(results_list)}\nAvg Relevancy: {avg_rel:.4f}\nAvg Correctness: {avg_corr:.4f}\n"
        with open(summary_path, 'w', encoding='utf-8') as f: f.write(summary_text)
        print(f"\n[DONE] Results saved to: {session_dir}")

if __name__ == "__main__":
    main()