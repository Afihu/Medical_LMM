"""
test_single_eval.py
-------------------
Test evaluation on the ONE case for now.
"""

from evaluate import RAGEvaluator

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # Evaluate just one for now
    results = evaluator.evaluate_all_responses(max_samples=1)
    
    if results:
        print("✓ Evaluation successful!\n")
        report = evaluator.generate_report()
        print(report)
    else:
        print("❌ Evaluation failed")