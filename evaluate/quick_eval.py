"""
Quick Batch Evaluation Launcher
--------------------------------
Simple wrapper to run batch evaluation with common options.

Usage:
    # Evaluate all test cases
    python evaluate/quick_eval.py

    # Evaluate only first 5 cases
    python evaluate/quick_eval.py --limit 5

    # Use existing diagnosis files (skip diagnosis step)
    python evaluate/quick_eval.py --skip-diagnosis
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.run_batch_evaluation import BatchEvaluator
import argparse

def main():
    parser = argparse.ArgumentParser(description="Quick batch evaluation launcher")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--skip-diagnosis", action="store_true", help="Use existing diagnosis files")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════╗
║     Medical_LMM Batch Evaluation Pipeline               ║
║                                                          ║
║  This script will:                                       ║
║  1. Run diagnosis on all test cases                     ║
║  2. Match with ground truth                             ║
║  3. Evaluate with RAGAS metrics                         ║
║  4. Generate JSON + CSV + Markdown reports              ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        evaluator = BatchEvaluator(skip_diagnosis=args.skip_diagnosis)
        evaluator.run_batch_evaluation(limit=args.limit)
        print("\n🎉 Evaluation pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
