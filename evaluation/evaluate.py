"""
evaluate.py
-----------
Evaluation for Medical LMM RAG pipeline.

Evaluates 3 metrics on actual pipeline output:
1. Context Relevance: Are retrieved cases relevant to the question?
2. Answer Faithfulness: Does Gemini's answer stick to the retrieved cases?
3. Answer Relevance: Does the model's answer address the original question?

Usage:
  1. Run main_streamlit.py and use the web interface to generate test cases
  2. This script will automatically find responses and evaluate them
  3. Run: uv run python evaluation/run_evaluation.py
"""

import os
import json
import glob
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# --- Path configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESPONSES_FOLDER = os.path.join(PROJECT_ROOT, "responses")
CASES_FOLDER_TEXT = os.path.join(PROJECT_ROOT, "cases_text")
CASES_FOLDER_IMAGE = os.path.join(PROJECT_ROOT, "cases_image")


class RAGEvaluator:
    """Evaluate RAG pipeline on 3 metrics."""
    
    def __init__(self):
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")
        self.results = []
    
    def extract_qa_from_markdown(self, md_file):
        """
        Extract user question and model answer from response markdown file.
        
        Returns: (user_question, ai_answer) or (None, None) if parse fails
        """
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            lines = content.split("\n")
            user_q = None
            ai_ans = None
            
            current_section = None
            user_lines = []
            ai_lines = []
            
            for line in lines:
                if line.startswith("**User:**"):
                    if current_section == "ai":
                        ai_ans = "\n".join(ai_lines).strip()
                    current_section = "user"
                    user_lines = [line.replace("**User:**", "").strip()]
                elif line.startswith("**AI:**"):
                    if current_section == "user":
                        user_q = "\n".join(user_lines).strip()
                    current_section = "ai"
                    ai_lines = [line.replace("**AI:**", "").strip()]
                else:
                    if current_section == "user":
                        user_lines.append(line)
                    elif current_section == "ai":
                        ai_lines.append(line)
            
            # Capture final answer
            if current_section == "ai":
                ai_ans = "\n".join(ai_lines).strip()
            
            return user_q, ai_ans
        except Exception as e:
            print(f"  Error parsing {os.path.basename(md_file)}: {e}")
            return None, None
    
    def load_retrieved_cases(self):
        """
        Load retrieved cases from /cases_text/ and /cases_image/ folders.
        These are the cases that were retrieved by Qdrant during the last query.
        
        Returns: Formatted string of all retrieved cases
        """
        cases_list = []
        
        # Load text cases
        if os.path.exists(CASES_FOLDER_TEXT):
            for filename in sorted(os.listdir(CASES_FOLDER_TEXT)):
                if filename.endswith(".json"):
                    path = os.path.join(CASES_FOLDER_TEXT, filename)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            case_data = json.load(f)
                        cases_list.append(("text", case_data))
                    except Exception as e:
                        print(f"  Error loading {filename}: {e}")
        
        # Load image cases
        if os.path.exists(CASES_FOLDER_IMAGE):
            for filename in sorted(os.listdir(CASES_FOLDER_IMAGE)):
                if filename.endswith(".json"):
                    path = os.path.join(CASES_FOLDER_IMAGE, filename)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            case_data = json.load(f)
                        cases_list.append(("image", case_data))
                    except Exception as e:
                        print(f"  Error loading {filename}: {e}")
        
        # Format as readable text
        formatted_cases = []
        for case_type, case_data in cases_list:
            case_id = case_data.get("id", "unknown")
            score = case_data.get("score", "N/A")
            payload = case_data.get("payload", {})
            
            case_str = f"Case ({case_type}): ID={case_id}, Similarity Score={score}\n"
            for key, value in payload.items():
                case_str += f"  {key}: {value}\n"
            
            formatted_cases.append(case_str)
        
        if not formatted_cases:
            return "No cases retrieved"
        
        return "\n".join(formatted_cases)
    
    def evaluate_sample(self, user_question, retrieved_cases, gemini_answer):
        """
        Evaluate ONE sample on 3 metrics.
        
        Args:
            user_question: Original question from user
            retrieved_cases: Cases retrieved from Qdrant (as formatted text)
            gemini_answer: Gemini's diagnostic response
        
        Returns:
            Dict with 3 metric scores (0-1)
        """
        
        # Build evaluation prompt
        eval_prompt = f"""You are evaluating a medical AI diagnostic system that uses RAG (Retrieval-Augmented Generation).

USER'S ORIGINAL QUESTION:
{user_question}

RETRIEVED CASES (these are what the system found in Qdrant):
{retrieved_cases}

AI'S DIAGNOSTIC ANSWER (this is what Gemini generated):
{gemini_answer}

Now evaluate the system on these 3 metrics. Rate each on a scale of 0-1:

1. CONTEXT RELEVANCE: How relevant are the retrieved cases to answering the user's question?
   0 = completely irrelevant, 1 = perfectly relevant

2. ANSWER FAITHFULNESS: Does the AI's answer stick ONLY to what's in the retrieved cases, or does it hallucinate/extrapolate beyond the cases?
   0 = lots of hallucinations/extrapolation, 1 = perfectly faithful to retrieved context

3. ANSWER RELEVANCE: Does the AI's answer actually address the user's original question?
   0 = off-topic or irrelevant, 1 = directly and fully addresses the question

RESPONSE FORMAT: Reply with ONLY 3 numbers separated by spaces (e.g., "0.9 0.85 0.95"), nothing else.
"""
        
        try:
            response = self.model.generate_content(eval_prompt)
            scores_text = response.text.strip()
            
            # Parse 3 numbers
            scores = [float(x) for x in scores_text.split()[:3]]
            
            return {
                "context_relevance": scores[0],
                "answer_faithfulness": scores[1],
                "answer_relevance": scores[2],
                "overall": sum(scores) / 3,
                "error": None
            }
        except Exception as e:
            print(f"  ⚠️  Evaluation error: {e}")
            return {
                "context_relevance": 0.5,
                "answer_faithfulness": 0.5,
                "answer_relevance": 0.5,
                "overall": 0.5,
                "error": str(e)
            }
    
    def evaluate_response_file(self, md_file):
        """
        Evaluate a single response markdown file.
        
        This will:
        1. Extract user question and AI answer from the markdown
        2. Load retrieved cases from /cases_text/ and /cases_image/
        3. Call evaluate_sample() to get 3 metrics
        4. Return results
        """
        print(f"Evaluating: {os.path.basename(md_file)}")
        
        # Extract Q&A from markdown
        user_q, ai_ans = self.extract_qa_from_markdown(md_file)
        if not user_q or not ai_ans:
            print(f"  ⚠️  Could not extract Q&A - skipping")
            return None
        
        # Load retrieved cases
        retrieved_cases = self.load_retrieved_cases()
        
        # Evaluate
        result = self.evaluate_sample(user_q, retrieved_cases, ai_ans)
        result["file"] = os.path.basename(md_file)
        result["user_question"] = user_q[:100] + "..." if len(user_q) > 100 else user_q
        
        return result
    
    def evaluate_all_responses(self, max_samples=None):
        """
        Find all response files and evaluate them.
        
        Args:
            max_samples: If set, only evaluate first N files (useful for testing)
        """
        response_files = sorted(glob.glob(os.path.join(RESPONSES_FOLDER, "*.md")))
        
        if not response_files:
            print(f"❌ No markdown files found in {RESPONSES_FOLDER}")
            return []
        
        if max_samples:
            response_files = response_files[:max_samples]
        
        print(f"Found {len(response_files)} response file(s)\n")
        
        self.results = []
        for i, md_file in enumerate(response_files, 1):
            result = self.evaluate_response_file(md_file)
            if result:
                self.results.append(result)
            print()
        
        return self.results
    
    def generate_report(self, output_path=None):
        """Generate evaluation report from results."""
        if not self.results:
            return "No evaluation results to report."
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(PROJECT_ROOT, f"eval_report_{timestamp}")
        
        # Calculate statistics
        context_scores = [r["context_relevance"] for r in self.results if r.get("error") is None]
        faith_scores = [r["answer_faithfulness"] for r in self.results if r.get("error") is None]
        relevance_scores = [r["answer_relevance"] for r in self.results if r.get("error") is None]
        overall_scores = [r["overall"] for r in self.results if r.get("error") is None]
        
        if not context_scores:
            return "No valid results to report."
        
        avg_context = sum(context_scores) / len(context_scores)
        avg_faith = sum(faith_scores) / len(faith_scores)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        avg_overall = sum(overall_scores) / len(overall_scores)
        
        # Build report
        report = "\n" + "="*80 + "\n"
        report += "MEDICAL LMM RAG PIPELINE EVALUATION REPORT\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "="*80 + "\n\n"
        
        report += "SUMMARY:\n"
        report += "-"*80 + "\n"
        report += f"Total samples evaluated: {len(self.results)}\n"
        report += f"Valid evaluations: {len(context_scores)}\n"
        report += f"Failed evaluations: {len(self.results) - len(context_scores)}\n\n"
        
        report += "METRIC RESULTS (Mean Scores 0-1):\n"
        report += "-"*80 + "\n"
        report += f"1. Context Relevance............. {avg_context:.4f}\n"
        report += f"   (Are retrieved cases relevant to the question?)\n"
        report += f"\n2. Answer Faithfulness.......... {avg_faith:.4f}\n"
        report += f"   (Does answer stick to retrieved cases?)\n"
        report += f"\n3. Answer Relevance............ {avg_relevance:.4f}\n"
        report += f"   (Does answer address the question?)\n"
        report += f"\nOVERALL SCORE.................. {avg_overall:.4f}\n"
        report += "\n"
        
        report += "INTERPRETATION GUIDE:\n"
        report += "-"*80 + "\n"
        report += """
Score Ranges:
  0.8-1.0: Excellent - System performing very well
  0.6-0.8: Good - System working, minor issues
  0.4-0.6: Fair - Needs improvement
  0.0-0.4: Poor - Significant problems

What to do if scores are low:

Context Relevance LOW (< 0.6):
  → Problem: Qdrant retrieval finding irrelevant cases
  → Fix: Check embedding quality, Qdrant tuning, vector dimension

Answer Faithfulness LOW (< 0.6):
  → Problem: Gemini hallucinating beyond retrieved cases
  → Fix: Improve prompt engineering, add constraints in prompt

Answer Relevance LOW (< 0.6):
  → Problem: Gemini's answers off-topic
  → Fix: Better prompt instructions, clearer question handling
"""
        
        report += "\n" + "="*80 + "\n"
        report += "PER-SAMPLE RESULTS:\n"
        report += "="*80 + "\n"
        
        for i, result in enumerate(self.results, 1):
            report += f"\nSample {i}: {result['file']}\n"
            report += f"  Question: {result['user_question']}\n"
            report += f"  Context Relevance: {result['context_relevance']:.4f}\n"
            report += f"  Answer Faithfulness: {result['answer_faithfulness']:.4f}\n"
            report += f"  Answer Relevance: {result['answer_relevance']:.4f}\n"
            report += f"  Overall: {result['overall']:.4f}\n"
            
            if result.get("error"):
                report += f"  ⚠️  Error: {result['error']}\n"
        
        # Save report
        report_file = f"{output_path}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"✓ Report saved to: {report_file}")
        
        return report


# --- MAIN USAGE ---
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # Evaluate all responses in /responses folder
    results = evaluator.evaluate_all_responses(max_samples=None)
    
    # Generate report
    if results:
        report = evaluator.generate_report()
        print("\n" + report)
    else:
        print("No evaluation results generated.")