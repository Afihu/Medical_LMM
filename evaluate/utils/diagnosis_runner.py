"""Diagnosis runner for batch evaluation."""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from scripts.llm_services.base import LLMProvider
from scripts.qdrant_services.query import run_query
from scripts.main_runtime.prompt_generate import generate_prompt
from scripts.embedding_generation_module.orchestrators import QueryOrchestrator
from evaluate.eval_config import EVAL_MODE, CONTEXT_TYPE


class DiagnosisRunner:
    """Handles running diagnosis on test cases with pluggable LLM providers."""
    
    def __init__(self, llm_provider: LLMProvider, output_dir: Path):
        """Initialize diagnosis runner.
        
        Args:
            llm_provider: LLMProvider instance (GeminiProvider, LMStudioProvider, etc.)
            output_dir: Directory to save diagnosis results
        """
        self.llm_provider = llm_provider
        self.output_dir = output_dir
    
    def run(self, case_id: str, prompt: str, diagnosis: Optional[str] = None) -> Optional[str]:
        """Run diagnosis on a single test case.
        
        Args:
            case_id: Test case identifier
            prompt: Patient query/symptoms text
            diagnosis: Ground truth diagnosis (used to find corresponding images)
            
        Returns:
            Path to saved diagnosis file, or None if failed
        """
        print(f"🔬 Running diagnosis for {case_id}...")
        
        try:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate embedding and query Qdrant (only if not in "internal" mode)
            retrieved_cases = {}
            
            if EVAL_MODE != "internal":
                # Query text collection
                text_vector = self._embed_query(prompt, session_id)
                text_cases = self._query_qdrant(text_vector, session_id)
                retrieved_cases.update(text_cases)
                
                # Query image collection if text+images mode
                if CONTEXT_TYPE == "text+images":
                    if diagnosis:
                        image_path = self._find_case_image(diagnosis)
                        if image_path:
                            print(f"   📸 Found image for {diagnosis}: {Path(image_path).name}")
                            image_vector = self._embed_query_image(image_path, session_id)
                            image_cases = self._query_qdrant_images(image_vector, session_id)
                            retrieved_cases.update(image_cases)
                        else:
                            print(f"   ℹ️  No images found for {diagnosis}, using text-only mode")
                    else:
                        print(f"   ⚠️  CONTEXT_TYPE is 'text+images' but no diagnosis provided, skipping image query")
                
                # Print retrieved cases once here
                self._print_retrieved_cases(retrieved_cases)
            else:
                print("[INFO] Internal mode: Skipping Qdrant retrieval (RAG disabled).")
            
            # Generate diagnosis using appropriate prompt template
            # eval_mode=None uses default prompt.txt, eval_mode="internal"/"rag"/"hybrid" use specific templates
            final_prompt, _ = generate_prompt(prompt, session_id=session_id, eval_mode=EVAL_MODE)
            ai_output = self._call_llm(final_prompt)
            
            # Save result
            output_path = self._save_diagnosis(case_id, prompt, retrieved_cases, 
                                              final_prompt, ai_output)
            
            print(f"✅ Diagnosis saved: {output_path.name}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error running diagnosis for {case_id}: {e}")
            return None
    
    def _find_case_image(self, diagnosis: str) -> Optional[str]:
        """Find first image for a diagnosis from test_cases/imgs/{diagnosis}/.
        
        Args:
            diagnosis: Diagnosis name (e.g., "Malaria", "Tuberculosis")
            
        Returns:
            Absolute path to first image file, or None if folder doesn't exist or is empty
        """
        # Get project root (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        imgs_dir = project_root / "test_cases" / "imgs" / diagnosis
        
        if not imgs_dir.exists():
            return None
        
        # Get first image file (jpg, png, jpeg)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files = list(imgs_dir.glob(ext))
            if image_files:
                return str(image_files[0].absolute())
        
        return None
    
    def _embed_query(self, prompt: str, session_id: str):
        """Generate text embedding."""
        with QueryOrchestrator(session_id=session_id) as orchestrator:
            return orchestrator.embed_text_query(prompt)
    
    def _embed_query_image(self, image_path: str, session_id: str):
        """Generate image embedding from image file.
        
        Args:
            image_path: Absolute path to image file
            session_id: Session identifier
            
        Returns:
            Image embedding vector
        """
        with QueryOrchestrator(session_id=session_id) as orchestrator:
            return orchestrator.embed_image_query(image_path)
    
    def _query_qdrant(self, text_vector, session_id: str) -> Dict:
        """Query Qdrant text collection for similar cases."""
        retrieved_cases, _ = run_query(
            text_vector=text_vector.tolist() if text_vector is not None else None,
            image_vector=None,
            top_k=3,
            session_id=session_id
        )
        return retrieved_cases if isinstance(retrieved_cases, dict) else {}
    
    def _query_qdrant_images(self, image_vector, session_id: str) -> Dict:
        """Query Qdrant image collection for similar cases.
        
        This reuses the image query functionality from the main pipeline.
        """
        retrieved_cases, _ = run_query(
            text_vector=None,
            image_vector=image_vector.tolist() if image_vector is not None else None,
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
        
        # Add instruction for RAG-only mode
        rag_instruction = ""
        if EVAL_MODE == "rag":
            rag_instruction = "Do not use your internal knowledge. Give diagnosis based on the retrieved cases only."

        simplified_prompt = f"""You are a medical diagnostician AI specializing in tropical and infectious diseases.

{rag_instruction}

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
    
    def _call_llm(self, prompt: str) -> Dict:
        """Call LLM provider for diagnosis.
        
        Supports any LLM provider (Gemini, LM Studio, etc.)
        
        Returns parsed JSON response.
        """
        # Build content for provider
        content = [{"text": prompt}]
        
        # Get provider name for logging
        provider_name = self.llm_provider.get_provider_name()
        print(f"\n🤖 {provider_name.title()} Response:")
        print("=" * 80)
        
        try:
            # Generate content using provider
            ai_text = self.llm_provider.generate_content(content)
            print(ai_text)
            print("=" * 80)
            print()
            
            # Parse response using provider
            ai_output = self.llm_provider.parse_response(ai_text)
            
            if "error" not in ai_output:
                print(f"✅ Successfully parsed JSON response")
            else:
                print(f"⚠️  Response parsing: {ai_output.get('error', 'Unknown error')}")
            
            return ai_output
            
        except Exception as e:
            print(f"❌ Error calling {provider_name.title()} provider: {e}")
            print("=" * 80)
            print()
            return {
                "raw_output": str(e),
                "error": f"Provider error: {str(e)}",
                "provider": provider_name
            }
    
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
