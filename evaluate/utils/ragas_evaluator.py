"""RAGAS evaluation utilities."""

import json
from typing import Dict, Any
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


class RAGASEvaluator:
    """Handles RAGAS evaluation of diagnosis results."""
    
    def __init__(self, api_key: str):
        """Initialize RAGAS evaluator."""
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-2.5-flash-lite",
            temperature=0,
            max_output_tokens=5130,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="abhinand/MedEmbed-base-v0.1")
    
    def evaluate_case(self, case_id: str, diag_file: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate a single case with RAGAS metrics."""
        print(f"📊 Evaluating {case_id} with RAGAS...")
        
        try:
            # Load and prepare data
            ragas_data = self._prepare_ragas_data(diag_file, ground_truth)
            
            # Build RAGAS sample
            sample = self._build_sample(ragas_data)
            dataset = EvaluationDataset(samples=[sample])
            
            # Run evaluation
            results = evaluate(
                dataset,
                metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
                llm=self.llm,
                embeddings=self.embeddings,
            )
            
            # Extract scores
            scores = self._extract_scores(results)
            
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
            "answer": self._flatten_answer(ai_response),
            "ground_truth": ground_truth
        }
    
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
    
    def _build_sample(self, ragas_data: Dict[str, Any]) -> SingleTurnSample:
        """Build RAGAS sample from prepared data."""
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
            response=ragas_data["answer"],
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
