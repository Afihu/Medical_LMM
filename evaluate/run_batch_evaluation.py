"""
Automated Batch Evaluation Script for Medical_LMM
--------------------------------------------------
Automates diagnosis and RAGAS evaluation pipeline with provider-agnostic LLM support.

Usage:
    python evaluate/run_batch_evaluation.py [--limit N] [--skip-diagnosis]

Configuration:
    Set LLM_PROVIDER environment variable or in evaluate/config.py:
    - "gemini": Google Gemini API (requires GEMINI_API_KEY)
    - "lmstudio": Local LM Studio or compatible server (requires LOCAL_LLM_URL)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Import configuration and LLM factory
from evaluate.eval_config import get_config, print_config
from scripts.llm_services.factory import get_llm_provider

# Import local utilities
from evaluate.utils.diagnosis_runner import DiagnosisRunner
from evaluate.utils.ragas_evaluator import RAGASEvaluator
from evaluate.utils.report_generator import ReportGenerator


class BatchEvaluator:
    """Main batch evaluation orchestrator with provider-agnostic LLM support."""
    
    def __init__(self, skip_diagnosis: bool = False):
        """Initialize batch evaluator.
        
        Loads configuration from evaluate/config.py and environment variables.
        Initializes LLM provider based on configuration.
        """
        load_dotenv()
        
        # Load and display configuration
        print("\n" + "=" * 60)
        print("📋 Loading Evaluation Configuration...")
        print("=" * 60)
        
        try:
            config = get_config()
            print_config()
        except ValueError as e:
            raise ValueError(f"❌ Configuration error: {e}")
        
        # Setup LLM provider from factory
        try:
            # Initialize provider based on evaluate/config.py settings
            # This overrides the default behavior which uses scripts/config/llm_config.py
            provider_type = config['llm_provider']
            self.llm_provider = get_llm_provider(provider_type)
            
            # Configure provider with specific settings
            if provider_type == "gemini":
                api_key = os.getenv("GEMINI_API_KEY")
                self.llm_provider.setup(
                    api_key=api_key,
                    model_name=config['gemini_model']
                )
            elif provider_type == "lmstudio":
                self.llm_provider.setup(
                    url=config['local_llm_url'],
                    model_name=config['lmstudio_model'],
                    temperature=config['lmstudio_temperature'],
                    max_tokens=config['lmstudio_max_tokens'],
                    top_p=config['lmstudio_top_p']
                )
                
            provider_name = self.llm_provider.get_provider_name()
            print(f"✅ LLM Provider initialized: {provider_name.upper()}")
        except Exception as e:
            raise ValueError(f"❌ Failed to initialize LLM provider: {e}")
        
        self.skip_diagnosis = skip_diagnosis
        self.base_dir = PROJECT_ROOT
        self.test_cases_path = self.base_dir / "test_cases" / "augmented_test.json"
        self.diagnosed_cases_dir = self.base_dir / "diagnosed_cases"
        self.results_dir = self.base_dir / "evaluate" / "batch_results"
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Load test cases
        try:
            with open(self.test_cases_path, 'r', encoding='utf-8') as f:
                self.test_cases = json.load(f)
            print(f"✅ Loaded {len(self.test_cases)} test cases from {self.test_cases_path.name}")
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Failed to parse {self.test_cases_path}: {e}")
        except Exception as e:
            raise ValueError(f"❌ Failed to load test cases from {self.test_cases_path}: {e}")
        
        # Initialize components with LLM provider
        self.diagnosis_runner = DiagnosisRunner(self.llm_provider, self.diagnosed_cases_dir)
        self.evaluator = RAGASEvaluator(self.llm_provider)
        
        print(f"✅ Initialized diagnosis runner and evaluator\n")
    
    def run_batch_evaluation(self, limit: Optional[int] = None, skip: int = 0):
        """Run complete batch evaluation pipeline.
        
        Args:
            limit: Maximum number of cases to evaluate
            skip: Number of cases to skip from the beginning
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create session-specific folder
        session_dir = self.results_dir / f"session_{timestamp}"
        session_dir.mkdir(exist_ok=True)
        
        results_file = session_dir / f"evaluation_results_{timestamp}.json"
        csv_file = session_dir / f"evaluation_results_{timestamp}.csv"
        report_file = session_dir / f"evaluation_report_{timestamp}.md"
        
        # Apply skip and limit
        test_cases = self.test_cases[skip:]
        if limit:
            test_cases = test_cases[:limit]
        
        self._print_header(len(test_cases))
        
        all_results = []
        for idx, case in enumerate(test_cases, 1):
            result = self._process_case(case, idx, len(test_cases))
            all_results.append(result)
            
            # Save intermediate results
            ReportGenerator.save_json(all_results, results_file)
            ReportGenerator.save_csv(all_results, csv_file)
            
            # Longer delay to avoid API quota exhaustion (especially for free tier)
            # Free tier has very limited requests per minute
            if idx < len(test_cases):  # Don't delay after last case
                print(f"   ⏳ Waiting 30 seconds before next case to respect API limits...")
                time.sleep(30)
        
        # Generate final report
        ReportGenerator.generate_markdown(all_results, report_file)
        
        self._print_summary(results_file, csv_file, report_file)
    
    def _process_case(self, case: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
        """Process a single test case."""
        case_id = case['id']
        print(f"\n[{idx}/{total}] Processing {case_id}...")
        
        case_result = {
            "case_id": case_id,
            "diagnosis": case['diagnosis'],
            "prompt": case['prompt'][:100] + "...",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Get diagnosis file
            diag_file = self._get_diagnosis_file(case)
            if not diag_file:
                case_result["status"] = "diagnosis_failed"
                return case_result
            
            case_result["diagnosis_file"] = diag_file
            
            # Step 2: Evaluate with RAGAS
            scores = self.evaluator.evaluate_case(case_id, diag_file, case['diagnosis'])
            
            # Extract evaluation errors if present
            evaluation_errors = scores.pop("evaluation_errors", None)
            
            case_result["scores"] = scores
            case_result["status"] = "completed"
            
            if evaluation_errors:
                case_result["evaluation_errors"] = evaluation_errors
            
        except Exception as e:
            print(f"❌ Error processing {case_id}: {e}")
            case_result["status"] = "error"
            case_result["error"] = str(e)
        
        return case_result
    
    def _get_diagnosis_file(self, case: Dict[str, Any]) -> Optional[str]:
        """Get diagnosis file (run new or use existing)."""
        if not self.skip_diagnosis:
            # Pass diagnosis to enable image discovery for text+images mode
            return self.diagnosis_runner.run(
                case['id'], 
                case['prompt'],
                diagnosis=case.get('diagnosis')  # Ground truth diagnosis for image lookup
            )
        
        # Find most recent diagnosis file
        diag_files = sorted(
            self.diagnosed_cases_dir.glob("diagnosis_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not diag_files:
            print(f"⚠️  No diagnosis files found, skipping {case['id']}")
            return None
        
        return str(diag_files[0])
    
    def _print_header(self, num_cases: int):
        """Print evaluation header."""
        print(f"\n{'='*60}")
        print(f"🚀 Starting Batch Evaluation")
        print(f"   Test cases: {num_cases}")
        print(f"   Skip diagnosis: {self.skip_diagnosis}")
        print(f"{'='*60}\n")
    
    def _print_summary(self, json_file: Path, csv_file: Path, report_file: Path):
        """Print completion summary."""
        session_folder = json_file.parent
        print(f"\n{'='*60}")
        print(f"✅ Batch Evaluation Complete!")
        print(f"   Session folder: {session_folder}")
        print(f"   - JSON results: {json_file.name}")
        print(f"   - CSV results: {csv_file.name}")
        print(f"   - Report: {report_file.name}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation on Medical_LMM test cases")
    parser.add_argument("--limit", type=int, help="Limit number of test cases to evaluate")
    parser.add_argument("--skip-diagnosis", action="store_true", help="Skip diagnosis step, use existing files")
    
    args = parser.parse_args()
    
    evaluator = BatchEvaluator(skip_diagnosis=args.skip_diagnosis)
    evaluator.run_batch_evaluation(limit=args.limit)


if __name__ == "__main__":
    main()
