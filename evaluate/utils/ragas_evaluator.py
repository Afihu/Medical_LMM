"""RAGAS evaluation utilities."""

import json
import time
import warnings
import traceback
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Import centralized model config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.config.model_config import RAGAS_EVALUATION_MODEL

# Suppress gRPC async cleanup warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='threading')
warnings.filterwarnings('ignore', message='.*InterceptedCall.*')


class RAGASEvaluator:
    """Handles RAGAS evaluation of diagnosis results."""
    
    def __init__(self, api_key: str):
        """Initialize RAGAS evaluator."""
        self.api_key = api_key
        # NOTE: This LLM is not actually used - ragas_worker.py runs evaluation in subprocess
        # Kept for backward compatibility
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=RAGAS_EVALUATION_MODEL,  # From centralized config
            temperature=0,
            max_output_tokens=5130,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="abhinand/MedEmbed-base-v0.1")
        self.worker_path = Path(__file__).parent / "ragas_worker.py"
    
    def _evaluate_in_subprocess(self, samples: List[Dict], metrics: List[str]) -> Optional[Dict[str, float]]:
        """Run RAGAS evaluation in isolated subprocess to avoid event loop issues.
        
        Args:
            samples: List of sample dicts with user_input, retrieved_contexts, response, reference
            metrics: List of metric names to evaluate
        
        Returns:
            Dict of metric scores, or None if subprocess fails
        """
        try:
            # Prepare input data
            input_data = {
                'api_key': self.api_key,
                'samples': samples,
                'metrics': metrics
            }
            input_json = json.dumps(input_data)
            
            # Run worker in subprocess
            result = subprocess.run(
                [sys.executable, str(self.worker_path)],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Always print stderr if it contains anything meaningful
            if result.stderr and len(result.stderr.strip()) > 0:
                # Check for quota errors first
                if 'ResourceExhausted: 429' in result.stderr or 'exceeded your current quota' in result.stderr:
                    print(f"   🚨 API QUOTA EXCEEDED!")
                    print(f"   Your Google Gemini API free tier quota is exhausted.")
                    print(f"   Solutions:")
                    print(f"      1. Wait for quota reset (usually midnight Pacific Time)")
                    print(f"      2. Upgrade to paid tier at https://ai.google.dev/")
                    print(f"      3. Use a different API key")
                    return None
                
                # Filter out gRPC noise
                stderr_lines = [line for line in result.stderr.split('\n') 
                               if line.strip() and 'ALTS creds ignored' not in line 
                               and 'alts_credentials.cc' not in line]
                if stderr_lines:
                    print(f"   ⚠️  Subprocess stderr:")
                    for line in stderr_lines[:10]:  # Limit to first 10 lines
                        print(f"      {line}")
            
            if result.returncode != 0:
                print(f"   ⚠️  Subprocess failed with code {result.returncode}")
                return None
            
            # Parse output
            try:
                output_data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Failed to parse subprocess output: {e}")
                print(f"   Stdout: {result.stdout[:500]}")
                return None
            
            if not output_data.get('success'):
                error = output_data.get('error', 'Unknown error')
                print(f"   ⚠️  Evaluation failed: {error}")
                return None
            
            scores = output_data.get('scores', {})
            # Debug: Check if scores are actually None/empty
            if not scores or all(v is None for v in scores.values()):
                print(f"   ⚠️  Warning: Subprocess returned empty or null scores")
                print(f"   Raw output: {output_data}")
            return scores
            
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  Subprocess timeout (300s)")
            return None
        except Exception as e:
            print(f"   ⚠️  Subprocess error: {e}")
            return None
    
    def _evaluate_with_retry(self, dataset: EvaluationDataset, metrics: list, 
                            metric_name: str, max_retries: int = 2) -> Optional[Dict[str, float]]:
        """Evaluate with retry logic and error logging.
        
        Args:
            dataset: RAGAS evaluation dataset
            metrics: List of metrics to evaluate
            metric_name: Human-readable name for logging
            max_retries: Maximum number of retry attempts (default: 2 = 1 initial + 1 retry)
        
        Returns:
            Dict of metric scores, or None if all attempts fail
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s...
                    print(f"   ⏳ Retry {attempt}/{max_retries-1} after {wait_time}s...")
                    time.sleep(wait_time)
                
                results = evaluate(
                    dataset,
                    metrics=metrics,
                    llm=self.llm,
                    embeddings=self.embeddings,
                )
                return self._extract_scores(results)
                
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                error_msg = str(e)[:200]  # Truncate long errors
                print(f"   ⚠️  {metric_name} attempt {attempt+1} failed: {error_type}: {error_msg}")
                
                # Don't retry on certain fatal errors
                if "API key" in error_msg or "authentication" in error_msg.lower():
                    print(f"   ❌ Fatal error, not retrying")
                    break
        
        # All retries exhausted
        print(f"   ❌ {metric_name} failed after {max_retries} attempts")
        print(f"   📋 Last error: {type(last_error).__name__}: {str(last_error)[:300]}")
        
        # Store full traceback for debugging
        full_trace = ''.join(traceback.format_exception(type(last_error), last_error, last_error.__traceback__))
        print(f"   📋 Full traceback:\n{full_trace[:500]}...")  # Print first 500 chars
        
        return None
    
    def evaluate_case(self, case_id: str, diag_file: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate a single case with RAGAS metrics."""
        print(f"📊 Evaluating {case_id} with RAGAS...")
        
        try:
            # Load and prepare data
            ragas_data = self._prepare_ragas_data(diag_file, ground_truth)
            
            # Build three samples: 
            # 1. Full answer (for context metrics + faithfulness)
            # 2. Concise answer (for answer relevancy)
            # 3. Diagnosis only (for answer correctness)
            sample_full = self._build_sample(ragas_data, use_concise=False)
            sample_concise = self._build_sample(ragas_data, use_concise=True)
            sample_diagnosis = self._build_sample(ragas_data, use_diagnosis_only=True)
            
            scores = {}
            error_details = []
            
            # Evaluate context metrics + faithfulness using subprocess (more reliable)
            print(f"   Evaluating context metrics + faithfulness in subprocess...")
            sample_full_dict = {
                'user_input': sample_full.user_input,
                'retrieved_contexts': sample_full.retrieved_contexts,
                'response': sample_full.response,
                'reference': sample_full.reference
            }
            results_full = self._evaluate_in_subprocess(
                samples=[sample_full_dict],
                metrics=['context_precision', 'context_recall', 'faithfulness']
            )
            
            if results_full and any(v is not None for v in results_full.values()):
                scores.update(results_full)
                print(f"   ✓ Context metrics complete: {results_full}")
            else:
                print(f"   ❌ Context metrics returned None or empty: {results_full}")
                scores.update({
                    "context_precision": None,
                    "context_recall": None,
                    "faithfulness": None,
                })
                error_details.append("Context metrics evaluation failed in subprocess")
            
            # Evaluate answer_relevancy with concise answer using subprocess
            print(f"   Evaluating answer relevancy with concise answer in subprocess...")
            sample_concise_dict = {
                'user_input': sample_concise.user_input,
                'retrieved_contexts': sample_concise.retrieved_contexts,
                'response': sample_concise.response,
                'reference': sample_concise.reference
            }
            results_concise = self._evaluate_in_subprocess(
                samples=[sample_concise_dict],
                metrics=['answer_relevancy']
            )
            
            if results_concise and results_concise.get("answer_relevancy") is not None:
                scores["answer_relevancy"] = results_concise.get("answer_relevancy")
                print(f"   ✓ Answer relevancy complete: {results_concise}")
            else:
                print(f"   ❌ Answer relevancy returned None or empty: {results_concise}")
                scores["answer_relevancy"] = None
                error_details.append("Answer relevancy evaluation failed in subprocess")
            
            # Evaluate answer_correctness with diagnosis-only answer using subprocess
            print(f"   Evaluating answer correctness with diagnosis-only answer in subprocess...")
            sample_diagnosis_dict = {
                'user_input': sample_diagnosis.user_input,
                'retrieved_contexts': sample_diagnosis.retrieved_contexts,
                'response': sample_diagnosis.response,
                'reference': sample_diagnosis.reference
            }
            results_correctness = self._evaluate_in_subprocess(
                samples=[sample_diagnosis_dict],
                metrics=['answer_correctness']
            )
            
            if results_correctness and results_correctness.get("answer_correctness") is not None:
                scores["answer_correctness"] = results_correctness.get("answer_correctness")
                print(f"   ✓ Answer correctness complete: {results_correctness}")
            else:
                print(f"   ❌ Answer correctness returned None or empty: {results_correctness}")
                scores["answer_correctness"] = None
                error_details.append("Answer correctness evaluation failed in subprocess")
            
            print(f"✅ Evaluation complete for {case_id}")
            print(f"   Scores: {scores}")
            
            # Add error details to scores if any metrics failed
            if error_details:
                scores["evaluation_errors"] = "; ".join(error_details)
            
            return scores
            
        except Exception as e:
            print(f"❌ Error evaluating {case_id}: {e}")
            import traceback
            print(f"   Full traceback:")
            traceback.print_exc()
            return {
                "context_precision": float('nan'),
                "context_recall": float('nan'),
                "faithfulness": float('nan'),
                "answer_relevancy": float('nan'),
                "answer_correctness": float('nan'),
                "error": str(e)
            }
    
    def _prepare_ragas_data(self, diag_file: str, ground_truth: str) -> Dict[str, Any]:
        """Load diagnosis file and prepare RAGAS data."""
        with open(diag_file, 'r', encoding='utf-8') as f:
            diag = json.load(f)
        
        prompt = diag.get('user_query', {}).get('text', '')
        ai_response = diag.get('ai_response', {})
        retrieved_cases = diag.get('retrieved_cases', {})
        
        retrieved_contexts = [case_data['text'] for case_data in retrieved_cases.values()] \
                            if retrieved_cases else []
        
        return {
            "question": prompt,
            "retrieved_contexts": retrieved_contexts,
            "answer_full": self._flatten_answer(ai_response),
            "answer_concise": self._extract_concise_answer(ai_response),
            "answer_diagnosis_only": self._extract_diagnosis_only(ai_response),
            "ai_response": ai_response,
            "ground_truth": ground_truth
        }
    
    def _extract_concise_answer(self, ai_response) -> str:
        """Extract concise answer for relevancy scoring.
        
        Priority:
        1. Use explicit 'concise_answer' field if present
        2. Extract top diagnosis + first evidence from differential_diagnosis
        3. Fallback to first diagnosis only
        """
        if isinstance(ai_response, str):
            return ai_response[:200]  # Truncate string responses
        
        if not isinstance(ai_response, dict):
            return str(ai_response)[:200]
        
        # Priority 1: Use explicit concise_answer field
        if "concise_answer" in ai_response:
            return str(ai_response["concise_answer"])
        
        # Priority 2: Extract from differential_diagnosis
        if "differential_diagnosis" in ai_response:
            diff_diagnosis = ai_response.get("differential_diagnosis", [])
            if diff_diagnosis and isinstance(diff_diagnosis, list) and len(diff_diagnosis) > 0:
                top = diff_diagnosis[0]
                if isinstance(top, dict):
                    disease = top.get("disease", "Unknown")
                    likelihood = top.get("likelihood", "")
                    reasoning = top.get("reasoning", "")
                    
                    # Extract first sentence or first 150 chars of reasoning
                    reasoning_short = reasoning.split('.')[0] if reasoning else ""
                    if len(reasoning_short) > 150:
                        reasoning_short = reasoning_short[:150] + "..."
                    
                    return f"Diagnosis: {disease} (Likelihood: {likelihood}). {reasoning_short}"
        
        # Priority 3: Extract from most_likely_diagnoses (alternative format)
        if "most_likely_diagnoses" in ai_response:
            diagnoses = ai_response.get("most_likely_diagnoses", [])
            if diagnoses and isinstance(diagnoses, list) and len(diagnoses) > 0:
                top = diagnoses[0]
                if isinstance(top, dict):
                    diagnosis = top.get("diagnosis", "Unknown")
                    rationale = top.get("rationale", "")
                    return f"Diagnosis: {diagnosis}. {rationale}"
        
        # Fallback: Return truncated full flatten
        return self._flatten_answer(ai_response)[:200]
    
    def _extract_diagnosis_only(self, ai_response) -> str:
        """Extract ONLY the diagnosis name for answer correctness comparison.
        
        Extracts just the disease name from concise_answer before (likelihood) or other metadata.
        Example: "Syphilis in Pregnancy (high likelihood) - ..." -> "Syphilis in Pregnancy"
        """
        if isinstance(ai_response, str):
            # If plain string, try to extract before first parenthesis
            if '(' in ai_response:
                return ai_response.split('(')[0].strip()
            return ai_response.strip()
        
        if not isinstance(ai_response, dict):
            return str(ai_response)
        
        # Priority 1: Extract from explicit 'concise_answer' field
        if "concise_answer" in ai_response:
            concise = str(ai_response["concise_answer"])
            # Extract text before (likelihood) or first hyphen
            if '(' in concise:
                diagnosis = concise.split('(')[0].strip()
            elif ' - ' in concise:
                diagnosis = concise.split(' - ')[0].strip()
            else:
                diagnosis = concise.strip()
            return diagnosis
        
        # Priority 2: Extract from differential_diagnosis
        if "differential_diagnosis" in ai_response:
            diff_diagnosis = ai_response.get("differential_diagnosis", [])
            if diff_diagnosis and isinstance(diff_diagnosis, list) and len(diff_diagnosis) > 0:
                top = diff_diagnosis[0]
                if isinstance(top, dict):
                    return top.get("disease", "Unknown")
        
        # Priority 3: Extract from most_likely_diagnoses (alternative format)
        if "most_likely_diagnoses" in ai_response:
            diagnoses = ai_response.get("most_likely_diagnoses", [])
            if diagnoses and isinstance(diagnoses, list) and len(diagnoses) > 0:
                top = diagnoses[0]
                if isinstance(top, dict):
                    return top.get("diagnosis", "Unknown")
        
        # Fallback: return "Unknown"
        return "Unknown"
    
    def _flatten_answer(self, ai_response) -> str:
        """Flatten AI response to string format for RAGAS.
        
        Handles differential_diagnosis format (current), legacy formats, and text strings.
        """
        # If already a string, return as-is
        if isinstance(ai_response, str):
            return ai_response
        
        # Handle non-dict types
        if not isinstance(ai_response, dict):
            return str(ai_response)
        
        output_parts = []
        
        # Handle CURRENT format: differential_diagnosis only
        if "differential_diagnosis" in ai_response:
            diff_diagnosis = ai_response.get("differential_diagnosis", [])
            if diff_diagnosis and isinstance(diff_diagnosis, list):
                output_parts.append("DIFFERENTIAL DIAGNOSIS:")
                for idx, entry in enumerate(diff_diagnosis, 1):
                    if isinstance(entry, dict):
                        disease = entry.get("disease", "Unknown")
                        reasoning = entry.get("reasoning", "")
                        likelihood = entry.get("likelihood", "")
                        
                        output_parts.append(f"\n{idx}. {disease} (Likelihood: {likelihood})")
                        if reasoning:
                            output_parts.append(f"   Reasoning: {reasoning}")
            
            if output_parts:
                return "\n".join(output_parts)
        
        # Handle most_likely_diagnoses format (alternative minimal format)
        if "most_likely_diagnoses" in ai_response:
            output_parts.append("DIFFERENTIAL DIAGNOSIS:")
            for item in ai_response.get("most_likely_diagnoses", []):
                if isinstance(item, dict):
                    rank = item.get("rank", "?")
                    diagnosis = item.get("diagnosis", "Unknown")
                    rationale = item.get("rationale", "")
                    evidence = item.get("supporting_evidence", {})
                    case_id = evidence.get("case_id", "") if isinstance(evidence, dict) else ""
                    snippet = evidence.get("snippet", "") if isinstance(evidence, dict) else ""
                    
                    output_parts.append(f"\n{rank}. {diagnosis}")
                    if rationale:
                        output_parts.append(f"   Rationale: {rationale}")
                    if case_id:
                        output_parts.append(f"   Supporting Case: {case_id}")
                    if snippet:
                        output_parts.append(f"   Evidence: {snippet}")
            
            if output_parts:
                return "\n".join(output_parts)
        
        # Handle LEGACY format with analysis
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
        """Flatten AI response to string for RAGAS."""
        if isinstance(ai_response, str):
            return ai_response
        if not isinstance(ai_response, dict):
            return str(ai_response)
        
        parts = []
        
        # Extract analysis
        if analysis := ai_response.get("analysis", []):
            parts.append("CLINICAL ANALYSIS:")
            for step in analysis:
                if isinstance(step, dict) and (step_name := step.get("step")) and (content := step.get("content")):
                    parts.append(f"\n{step_name}:")
                    parts.append(content)
        
        # Extract differential diagnosis
        if diff_diagnosis := ai_response.get("differential_diagnosis", []):
            parts.append("\n\nDIFFERENTIAL DIAGNOSIS:")
            for idx, entry in enumerate(diff_diagnosis, 1):
                if isinstance(entry, dict):
                    disease = entry.get("disease", "Unknown")
                    reasoning = entry.get("reasoning", "")
                    likelihood = entry.get("likelihood", "")
                    parts.append(f"\n{idx}. {disease} (Likelihood: {likelihood})")
                    if reasoning:
                        parts.append(f"   Reasoning: {reasoning}")
        
        return "\n".join(parts) if parts else str(ai_response)
    
    def _build_sample(self, ragas_data: Dict[str, Any], use_concise: bool = False, use_diagnosis_only: bool = False) -> SingleTurnSample:
        """Build RAGAS sample from prepared data.
        
        Args:
            ragas_data: Prepared RAGAS data
            use_concise: If True, use concise answer; otherwise use full answer
            use_diagnosis_only: If True, use only diagnosis name (for answer correctness)
        """
        if use_diagnosis_only:
            answer_text = ragas_data["answer_diagnosis_only"]
        elif use_concise:
            answer_text = ragas_data["answer_concise"]
        else:
            answer_text = ragas_data["answer_full"]
        
        return SingleTurnSample(
            user_input=ragas_data["question"],
            retrieved_contexts=[
                f"Title: {c.get('Title', '')}\n"
                f"History: {c.get('History', '')}\n"
                f"Findings: {c.get('Clinical_Findings', '')}\n"
                f"Discussion: {c.get('Discussion', '')}\n"
                f"Summary: {c.get('Summary_Box_First_Line', '')}"
                for c in ragas_data["retrieved_contexts"]
            ],
            response=answer_text,
            reference=ragas_data["ground_truth"],
        )
    
    def _extract_scores(self, results) -> Dict[str, float]:
        """Extract scores from RAGAS results."""
        try:
            df = results.to_pandas()
            return {
                "context_precision": float(df['context_precision'].iloc[0]) if 'context_precision' in df.columns else 0.0,
                "context_recall": float(df['context_recall'].iloc[0]) if 'context_recall' in df.columns else 0.0,
                "faithfulness": float(df['faithfulness'].iloc[0]) if 'faithfulness' in df.columns else 0.0,
                "answer_relevancy": float(df['answer_relevancy'].iloc[0]) if 'answer_relevancy' in df.columns else 0.0
            }
        except Exception as e:
            print(f"⚠️  Error parsing RAGAS results: {e}")
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0
            }
