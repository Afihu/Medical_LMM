"""Diagnosis runner for batch evaluation."""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import google.generativeai as genai

from scripts.qdrant_services.query import run_query
from scripts.main_runtime.prompt_generate import generate_prompt
from scripts.embedding_generation_module.orchestrators import QueryOrchestrator


class DiagnosisRunner:
    """Handles running diagnosis on test cases."""
    
    def __init__(self, api_key: str, model_name: str, output_dir: Path):
        """Initialize diagnosis runner."""
        self.api_key = api_key
        self.model_name = model_name
        self.output_dir = output_dir
        genai.configure(api_key=api_key)
    
    def run(self, case_id: str, prompt: str) -> Optional[str]:
        """Run diagnosis on a single test case.
        
        Returns:
            Path to saved diagnosis file, or None if failed
        """
        print(f"🔬 Running diagnosis for {case_id}...")
        
        try:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate embedding and query Qdrant
            text_vector = self._embed_query(prompt, session_id)
            retrieved_cases = self._query_qdrant(text_vector, session_id)
            
            # Print retrieved cases once here
            self._print_retrieved_cases(retrieved_cases)
            
            # Generate diagnosis using prompt.txt
            final_prompt, _ = generate_prompt(prompt, session_id=session_id)
            ai_output = self._call_gemini(final_prompt)
            
            # Save result
            output_path = self._save_diagnosis(case_id, prompt, retrieved_cases, 
                                              final_prompt, ai_output)
            
            print(f"✅ Diagnosis saved: {output_path.name}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error running diagnosis for {case_id}: {e}")
            return None
    
    def _embed_query(self, prompt: str, session_id: str):
        """Generate text embedding."""
        with QueryOrchestrator(session_id=session_id) as orchestrator:
            return orchestrator.embed_text_query(prompt)
    
    def _query_qdrant(self, text_vector, session_id: str) -> Dict:
        """Query Qdrant for similar cases."""
        retrieved_cases, _ = run_query(
            text_vector=text_vector.tolist() if text_vector is not None else None,
            image_vector=None,
            top_k=3,
            session_id=session_id
        )
        return retrieved_cases if isinstance(retrieved_cases, dict) else {}
    
    def _print_retrieved_cases(self, retrieved_cases: Dict):
        """Print retrieved cases information once."""
        print(f"\n📚 Retrieved {len(retrieved_cases)} cases:")
        for case_id, case_data in retrieved_cases.items():
            # Title is nested inside 'text' field
            text_data = case_data.get('text', {})
            title = text_data.get('Title', 'Unknown')
            summary = text_data.get('Summary_Box_First_Line', '')
            score = case_data.get('similarity_score', 0.0)
            
            print(f"   - Case {case_id}: {title}")
            print(f"     Diagnosis: {summary}")
            print(f"     Similarity: {score:.4f}")
        print()
    
    def _generate_prompt(self, prompt: str, session_id: str, retrieved_cases: Dict) -> str:
        """Generate simplified prompt for evaluation."""
        # Format retrieved cases with similarity scores
        cases_text = self._format_cases_for_prompt(retrieved_cases)
        
        simplified_prompt = f"""You are a medical diagnostician AI specializing in tropical and infectious diseases.

Patient Query: {prompt}

Retrieved Similar Cases:
{cases_text}

Required JSON Output Format (return ONLY valid JSON, no other text):

{{
  "most_likely_diagnoses": [
    {{
      "rank": 1,
      "diagnosis": "<Diagnosis Name>",
      "supporting_evidence": {{
        "case_id": "<ID>",
        "snippet": "<relevant quote from the case>"
      }},
      "rationale": "<one-line explanation>"
    }},
    {{
      "rank": 2,
      "diagnosis": "<Diagnosis Name>",
      "supporting_evidence": {{
        "case_id": "<ID>",
        "snippet": "<relevant quote from the case>"
      }},
      "rationale": "<one-line explanation>"
    }}
  ]
}}

Instructions:
- Provide 3-5 most likely diagnoses in ranked order (rank 1 = most likely)
- For each diagnosis, cite the most relevant retrieved case by case_id
- Include a brief snippet (quote) from that case supporting your diagnosis
- Keep rationale to ONE line explaining why this diagnosis fits the patient
- Output ONLY valid JSON following the exact schema above
- Do NOT include any text outside the JSON object
- Start with {{ and end with }}
"""
        return simplified_prompt
    
    def _format_cases_for_prompt(self, retrieved_cases: Dict) -> str:
        """Format retrieved cases for simplified prompt."""
        if not retrieved_cases:
            return "No similar cases retrieved."
        
        formatted = []
        for case_id, case_data in retrieved_cases.items():
            text_data = case_data.get('text', {})
            title = text_data.get('Title', 'Unknown')
            history = text_data.get('History', '')
            findings = text_data.get('Clinical_Findings', '')
            summary = text_data.get('Summary_Box_First_Line', '')
            
            formatted.append(f"Case {case_id}: {title}")
            if summary:
                formatted.append(f"  Diagnosis: {summary}")
            if history:
                formatted.append(f"  History: {history[:200]}...")
            if findings:
                formatted.append(f"  Findings: {findings[:200]}...")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _call_gemini(self, prompt: str) -> Dict:
        """Call Gemini API for diagnosis.
        
        Returns parsed JSON response.
        """
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content([{"text": prompt}])
        ai_text = response.text.strip()
        
        # Print Gemini response
        print(f"\n🤖 Gemini Response:")
        print("=" * 80)
        print(ai_text)
        print("=" * 80)
        print()
        
        # Extract JSON from code blocks if present
        match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', ai_text)
        if match:
            ai_text = match.group(1).strip()
        
        # Parse JSON
        try:
            parsed = json.loads(ai_text)
            print(f"✅ Successfully parsed JSON response")
            return parsed
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            print(f"    Attempting to extract JSON from text...")
            # Try to find JSON object in text
            json_match = re.search(r'\{[\s\S]*\}', ai_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    print(f"✅ Successfully extracted and parsed JSON")
                    return parsed
                except:
                    pass
            return {"raw_output": ai_text, "error": "Invalid JSON format"}
    
    def _save_diagnosis(self, case_id: str, prompt: str, retrieved_cases: Dict,
                       final_prompt: str, ai_output: Dict) -> Path:
        """Save diagnosis to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"diagnosis_{timestamp}.json"
        output_path = self.output_dir / filename
        
        diagnostic_record = {
            "timestamp": timestamp,
            "case_id": case_id,
            "user_query": {"text": prompt, "images": []},
            "has_image": False,
            "retrieved_cases": retrieved_cases,
            "generated_prompt": final_prompt,
            "ai_response": ai_output,
            "diagnosis": self._extract_diagnoses(ai_output),
            "correct": None
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(diagnostic_record, f, indent=4, ensure_ascii=False)
        
        return output_path
    
    def _extract_diagnoses(self, ai_output: Dict) -> list:
        """Extract diagnosis list from JSON output."""
        diagnoses = []
        
        # Handle differential_diagnosis format (current)
        if "differential_diagnosis" in ai_output:
            for item in ai_output.get("differential_diagnosis", []):
                if isinstance(item, dict) and "disease" in item:
                    diagnoses.append(item["disease"])
        # Handle legacy most_likely_diagnoses format
        elif "most_likely_diagnoses" in ai_output:
            for item in ai_output.get("most_likely_diagnoses", []):
                if isinstance(item, dict) and "diagnosis" in item:
                    diagnoses.append(item["diagnosis"])
        
        return diagnoses
