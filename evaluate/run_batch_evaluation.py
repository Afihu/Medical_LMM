"""
Automated Batch Evaluation Script for Medical_LMM
--------------------------------------------------
This script automates the entire evaluation pipeline:
1. Loads test cases from test_cases/cases.json
2. Runs diagnosis system on each case (via main runtime)
3. Matches diagnosed cases with ground truth (dataset.py logic)
4. Splits into individual RAGAS datasets
5. Evaluates each case with RAGAS metrics
6. Generates comprehensive evaluation report

Usage:
    python evaluate/run_batch_evaluation.py [--limit N] [--skip-diagnosis]
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Import your diagnosis system
from scripts.qdrant_services.query import run_query
from scripts.main_runtime.prompt_generate import generate_prompt
from scripts.embedding_generation_module.orchestrators import QueryOrchestrator
import google.generativeai as genai


class BatchEvaluator:
    def __init__(self, skip_diagnosis: bool = False):
        """Initialize the batch evaluator."""
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        if not self.api_key:
            raise ValueError("❌ Missing GEMINI_API_KEY in .env")
        
        genai.configure(api_key=self.api_key)
        
        self.skip_diagnosis = skip_diagnosis
        self.base_dir = PROJECT_ROOT
        self.test_cases_path = self.base_dir / "test_cases" / "cases.json"
        self.diagnosed_cases_dir = self.base_dir / "diagnosed_cases"
        self.results_dir = self.base_dir / "evaluate" / "batch_results"
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Load test cases
        with open(self.test_cases_path, 'r', encoding='utf-8') as f:
            self.test_cases = json.load(f)
        
        print(f"✅ Loaded {len(self.test_cases)} test cases")
        
    def run_diagnosis(self, case: Dict[str, Any]) -> str:
        """Run diagnosis system on a single test case."""
        case_id = case['id']
        prompt = case['prompt']
        
        print(f"\n🔬 Running diagnosis for {case_id}...")
        
        try:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate text embedding
            with QueryOrchestrator(session_id=session_id) as orchestrator:
                text_vector = orchestrator.embed_text_query(prompt)
            
            # Query Qdrant
            retrieved_cases, saved_dir = run_query(
                text_vector=text_vector.tolist() if text_vector is not None else None,
                image_vector=None,
                top_k=5,
                session_id=session_id
            )
            
            # Generate prompt
            final_prompt, decoded_images_path = generate_prompt(prompt, session_id=session_id)
            
            # Build content for Gemini
            content_parts = [{"text": final_prompt}]
            
            # Add reference images if any
            for img_path in decoded_images_path:
                try:
                    with open(img_path, "rb") as f:
                        data = f.read()
                    content_parts.append({
                        "inline_data": {"mime_type": "image/png", "data": data}
                    })
                except Exception as e:
                    print(f"⚠️  Could not attach image {img_path}: {e}")
            
            # Generate with Gemini
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(content_parts)
            ai_text = response.text.strip()
            
            # Parse JSON response
            import re
            match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', ai_text)
            if match:
                ai_text = match.group(1).strip()
            
            try:
                ai_output = json.loads(ai_text)
            except json.JSONDecodeError:
                print(f"⚠️  Invalid JSON from Gemini, saving raw text")
                ai_output = {"raw_output": ai_text, "error": "Invalid JSON format"}
            
            # Save diagnostic record
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"diagnosis_{timestamp}.json"
            output_path = self.diagnosed_cases_dir / filename
            
            diagnostic_record = {
                "timestamp": timestamp,
                "case_id": case_id,
                "user_query": {
                    "text": prompt,
                    "images": [],
                },
                "has_image": False,
                "retrieved_cases": retrieved_cases if isinstance(retrieved_cases, dict) else {},
                "generated_prompt": final_prompt,
                "ai_response": ai_output,
                "diagnosis": ai_output.get("differential_diagnosis", None),
                "correct": None
            }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(diagnostic_record, f, indent=4, ensure_ascii=False)
            
            print(f"✅ Diagnosis saved: {filename}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error running diagnosis for {case_id}: {e}")
            return None
    
    def flatten_answer(self, ai_response: Dict) -> str:
        """Flatten AI response to string format for RAGAS."""
        if isinstance(ai_response, str):
            return ai_response
        
        if not isinstance(ai_response, dict):
            return str(ai_response)
        
        output_parts = []
        
        # Extract analysis steps
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
        
        if not output_parts:
            return str(ai_response)
        
        return "\n".join(output_parts)
    
    def match_and_prepare_ragas_data(self, case: Dict[str, Any], diag_file: str) -> Dict[str, Any]:
        """Match diagnosed case with test case and prepare RAGAS dataset."""
        with open(diag_file, 'r', encoding='utf-8') as f:
            diag = json.load(f)
        
        prompt = diag.get('user_query', {}).get('text', '')
        ai_response = diag.get('ai_response', {})
        retrieved_cases = diag.get('retrieved_cases', {})
        
        # Prepare retrieved contexts
        retrieved_contexts = [case_data['text'] for case_data in retrieved_cases.values()] if retrieved_cases else []
        
        # Flatten answer
        answer_text = self.flatten_answer(ai_response)
        
        return {
            "question": prompt,
            "retrieved_contexts": retrieved_contexts,
            "answer": answer_text,
            "ground_truth": case['diagnosis']
        }
    
    def evaluate_case(self, ragas_data: Dict[str, Any], case_id: str) -> Dict[str, float]:
        """Evaluate a single case with RAGAS metrics."""
        print(f"📊 Evaluating {case_id} with RAGAS...")
        
        try:
            # Build sample
            sample = SingleTurnSample(
                user_input=ragas_data["question"],
                retrieved_contexts=[
                    f"Title: {c.get('Title', '')}\n"
                    f"History: {c.get('History', '')}\n"
                    f"Findings: {c.get('Clinical_Findings', '')}\n"
                    f"Discussion: {c.get('Discussion', '')}\n"
                    f"Summary: {c.get('Summary_Box_First_Line', '')}"
                    for c in ragas_data["retrieved_contexts"]
                ],
                response=ragas_data["answer"],
                reference=ragas_data["ground_truth"],
            )
            
            dataset = EvaluationDataset(samples=[sample])
            
            # Setup LLM and embeddings
            llm = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model="gemini-2.5-flash-lite",
                temperature=0,
                max_output_tokens=5130,
            )
            
            embeddings = HuggingFaceEmbeddings(model_name="abhinand/MedEmbed-base-v0.1")
            
            # Run evaluation
            results = evaluate(
                dataset,
                metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
                llm=llm,
                embeddings=embeddings,
            )
            
            # RAGAS 0.3.x returns EvaluationResult object with to_pandas() method
            # Convert to pandas DataFrame then extract scores
            try:
                df = results.to_pandas()
                scores = {
                    "context_precision": float(df['context_precision'].iloc[0]) if 'context_precision' in df.columns else 0.0,
                    "context_recall": float(df['context_recall'].iloc[0]) if 'context_recall' in df.columns else 0.0,
                    "faithfulness": float(df['faithfulness'].iloc[0]) if 'faithfulness' in df.columns else 0.0,
                    "answer_relevancy": float(df['answer_relevancy'].iloc[0]) if 'answer_relevancy' in df.columns else 0.0
                }
            except Exception as parse_error:
                print(f"⚠️  Error parsing RAGAS results: {parse_error}")
                print(f"   Results type: {type(results)}")
                print(f"   Results: {results}")
                scores = {
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0
                }
            
            print(f"✅ Evaluation complete for {case_id}")
            print(f"   Scores: {scores}")
            return scores
            
        except Exception as e:
            print(f"❌ Error evaluating {case_id}: {e}")
            return {
                "context_precision": None,
                "context_recall": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "error": str(e)
            }
    
    def run_batch_evaluation(self, limit: int = None):
        """Run complete batch evaluation pipeline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        csv_file = self.results_dir / f"evaluation_results_{timestamp}.csv"
        report_file = self.results_dir / f"evaluation_report_{timestamp}.md"
        
        all_results = []
        test_cases = self.test_cases[:limit] if limit else self.test_cases
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Batch Evaluation")
        print(f"   Test cases: {len(test_cases)}")
        print(f"   Skip diagnosis: {self.skip_diagnosis}")
        print(f"{'='*60}\n")
        
        for idx, case in enumerate(test_cases, 1):
            case_id = case['id']
            print(f"\n[{idx}/{len(test_cases)}] Processing {case_id}...")
            
            case_result = {
                "case_id": case_id,
                "diagnosis": case['diagnosis'],
                "prompt": case['prompt'][:100] + "...",
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                # Step 1: Run diagnosis (if not skipped)
                if not self.skip_diagnosis:
                    diag_file = self.run_diagnosis(case)
                    if not diag_file:
                        case_result["status"] = "diagnosis_failed"
                        all_results.append(case_result)
                        continue
                    case_result["diagnosis_file"] = diag_file
                else:
                    # Find most recent diagnosis file for this case
                    # This is a simplified version - you may need better matching logic
                    diag_files = sorted(self.diagnosed_cases_dir.glob("diagnosis_*.json"), 
                                      key=lambda x: x.stat().st_mtime, reverse=True)
                    if not diag_files:
                        print(f"⚠️  No diagnosis files found, skipping {case_id}")
                        case_result["status"] = "no_diagnosis_file"
                        all_results.append(case_result)
                        continue
                    diag_file = str(diag_files[0])
                    case_result["diagnosis_file"] = diag_file
                
                # Step 2: Prepare RAGAS data
                ragas_data = self.match_and_prepare_ragas_data(case, diag_file)
                
                # Step 3: Evaluate with RAGAS
                scores = self.evaluate_case(ragas_data, case_id)
                case_result["scores"] = scores
                case_result["status"] = "completed"
                
                # Small delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error processing {case_id}: {e}")
                case_result["status"] = "error"
                case_result["error"] = str(e)
            
            all_results.append(case_result)
            
            # Save intermediate results (JSON and CSV)
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            self.save_results_to_csv(all_results, csv_file)
        
        # Generate report
        self.generate_report(all_results, report_file)
        
        print(f"\n{'='*60}")
        print(f"✅ Batch Evaluation Complete!")
        print(f"   JSON results: {results_file}")
        print(f"   CSV results: {csv_file}")
        print(f"   Report: {report_file}")
        print(f"{'='*60}\n")
    
    def save_results_to_csv(self, results: List[Dict], csv_file: Path):
        """Save evaluation results to CSV file."""
        if not results:
            return
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'case_id',
                'diagnosis',
                'status',
                'context_precision',
                'context_recall',
                'faithfulness',
                'answer_relevancy',
                'timestamp',
                'error'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'case_id': result.get('case_id', ''),
                    'diagnosis': result.get('diagnosis', ''),
                    'status': result.get('status', ''),
                    'timestamp': result.get('timestamp', '')
                }
                
                # Add scores if available
                scores = result.get('scores', {})
                row['context_precision'] = scores.get('context_precision', '')
                row['context_recall'] = scores.get('context_recall', '')
                row['faithfulness'] = scores.get('faithfulness', '')
                row['answer_relevancy'] = scores.get('answer_relevancy', '')
                
                # Add error if present
                row['error'] = result.get('error', '')
                
                writer.writerow(row)
    
    def generate_report(self, results: List[Dict], report_file: Path):
        """Generate markdown evaluation report."""
        completed = [r for r in results if r.get("status") == "completed"]
        failed = [r for r in results if r.get("status") != "completed"]
        
        # Calculate averages
        avg_scores = {}
        if completed:
            for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
                scores = [r["scores"][metric] for r in completed if r["scores"].get(metric) is not None]
                avg_scores[metric] = sum(scores) / len(scores) if scores else 0
        
        # Generate markdown
        report = f"""# Medical_LMM Batch Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

- **Total Cases:** {len(results)}
- **Completed:** {len(completed)}
- **Failed:** {len(failed)}

## Average RAGAS Scores

| Metric | Score |
|--------|-------|
| Context Precision | {avg_scores.get('context_precision', 0):.4f} |
| Context Recall | {avg_scores.get('context_recall', 0):.4f} |
| Faithfulness | {avg_scores.get('faithfulness', 0):.4f} |
| Answer Relevancy | {avg_scores.get('answer_relevancy', 0):.4f} |

## Individual Case Results

"""
        
        for result in completed:
            scores = result.get("scores", {})
            report += f"""### {result['case_id']} - {result['diagnosis']}

**Scores:**
- Context Precision: {scores.get('context_precision', 'N/A'):.4f}
- Context Recall: {scores.get('context_recall', 'N/A'):.4f}
- Faithfulness: {scores.get('faithfulness', 'N/A'):.4f}
- Answer Relevancy: {scores.get('answer_relevancy', 'N/A'):.4f}

---

"""
        
        if failed:
            report += "\n## Failed Cases\n\n"
            for result in failed:
                report += f"- **{result['case_id']}**: {result.get('status', 'unknown')} - {result.get('error', 'N/A')}\n"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation on Medical_LMM test cases")
    parser.add_argument("--limit", type=int, help="Limit number of test cases to evaluate")
    parser.add_argument("--skip-diagnosis", action="store_true", help="Skip diagnosis step, use existing files")
    
    args = parser.parse_args()
    
    evaluator = BatchEvaluator(skip_diagnosis=args.skip_diagnosis)
    evaluator.run_batch_evaluation(limit=args.limit)


if __name__ == "__main__":
    main()
