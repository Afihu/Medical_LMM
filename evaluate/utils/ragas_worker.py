"""
Subprocess worker for RAGAS evaluation.
Runs in isolated process to avoid event loop conflicts.
"""

import sys
import json
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*coroutine.*was never awaited.*')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def run_evaluation(input_data):
    """Run RAGAS evaluation in isolated subprocess."""
    try:
        # Parse input
        api_key = input_data['api_key']
        samples = input_data['samples']
        metrics = input_data['metrics']
        
        # Initialize LLM and embeddings
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        
        # Build dataset
        eval_samples = []
        for sample_data in samples:
            eval_samples.append(SingleTurnSample(
                user_input=sample_data['user_input'],
                retrieved_contexts=sample_data['retrieved_contexts'],
                response=sample_data['response'],
                reference=sample_data.get('reference')
            ))
        
        dataset = EvaluationDataset(samples=eval_samples)
        
        # Map metric names to objects
        metric_map = {
            'context_precision': context_precision,
            'context_recall': context_recall,
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'answer_correctness': answer_correctness
        }
        metric_objs = [metric_map[m] for m in metrics if m in metric_map]
        
        # Run evaluation
        results = evaluate(
            dataset,
            metrics=metric_objs,
            llm=llm,
            embeddings=embeddings
        )
        
        # Extract scores
        scores = {}
        df = results.to_pandas()
        if not df.empty:
            for metric in metrics:
                if metric in df.columns:
                    value = df[metric].iloc[0]
                    # Convert to float, handle NaN
                    scores[metric] = float(value) if value == value else None
        
        return {
            'success': True,
            'scores': scores
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


if __name__ == '__main__':
    try:
        # Read input from stdin
        input_json = sys.stdin.read()
        input_data = json.loads(input_json)
        
        # Run evaluation
        result = run_evaluation(input_data)
        
        # Write result to stdout
        print(json.dumps(result))
        sys.exit(0)
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result))
        sys.exit(1)
