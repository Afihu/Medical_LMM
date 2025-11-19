"""
Subprocess worker for RAGAS evaluation.
Runs in isolated process to avoid event loop conflicts.
"""

import sys
import json
import warnings
import os
from pathlib import Path

# Suppress all warnings for clean console output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress TensorFlow and related warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (0=all, 1=info, 2=warning, 3=error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations message

# Suppress gRPC/ALTS warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Suppress gRPC messages
os.environ['GRPC_TRACE'] = ''  # Disable gRPC tracing

# Suppress HuggingFace warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformers warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'  # Disable advisory warnings
os.environ['HF_HOME'] = os.environ.get('TRANSFORMERS_CACHE', os.path.expanduser('~/.cache/huggingface'))
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # Disable HuggingFace progress bars

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from evaluate.config import RAGAS_EVALUATION_MODEL, RAGAS_EMBEDDINGS_MODEL

# Suppress absl logging after imports
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

# Suppress TensorFlow logging after imports
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    pass


def run_evaluation(input_data):
    """Run RAGAS evaluation in isolated subprocess."""
    try:
        # Parse input
        api_key = input_data.get('api_key')
        samples = input_data['samples']
        metrics = input_data['metrics']
        
        # Get provider configuration (defaults to Gemini for backward compatibility)
        ragas_llm_provider = input_data.get('ragas_llm_provider', 'gemini').lower()
        ragas_embeddings_provider = input_data.get('ragas_embeddings_provider', 'google').lower()
        ragas_embeddings_model = input_data.get('ragas_embeddings_model', 'abhinand/MedEmbed-base-v0.1')
        
        # Local LLM configuration (for lmstudio/local providers)
        local_llm_url = input_data.get('local_llm_url', 'http://localhost:1234')
        lmstudio_model = input_data.get('lmstudio_model', 'medgemma-4b-it')
        lmstudio_temperature = input_data.get('lmstudio_temperature', 0.0)
        lmstudio_max_tokens = input_data.get('lmstudio_max_tokens', 32768)
        
        # Initialize LLM based on provider
        if ragas_llm_provider == 'gemini':
            if not api_key:
                raise ValueError("RAGAS_LLM_PROVIDER is 'gemini' but api_key not provided")
            llm = ChatGoogleGenerativeAI(
                model=RAGAS_EVALUATION_MODEL,
                google_api_key=api_key,
                temperature=0
            )
        elif ragas_llm_provider in ['local', 'lmstudio']:
            # Use ChatOpenAI with custom base_url for OpenAI-compatible local servers
            # Works with LM Studio, vLLM, Ollama, etc.
            llm = ChatOpenAI(
                base_url=f"{local_llm_url}/v1",  # OpenAI-compatible endpoint
                api_key="dummy-key",  # Local servers often don't check API keys
                model=lmstudio_model,
                temperature=lmstudio_temperature,
                max_tokens=lmstudio_max_tokens,
            )
        else:
            raise ValueError(f"Unsupported RAGAS_LLM_PROVIDER: {ragas_llm_provider}")
        
        # Initialize embeddings based on provider
        if ragas_embeddings_provider == 'google':
            if not api_key:
                raise ValueError("RAGAS_EMBEDDINGS_PROVIDER is 'google' but api_key not provided")
            embeddings = GoogleGenerativeAIEmbeddings(
                model=RAGAS_EMBEDDINGS_MODEL,
                google_api_key=api_key
            )
        elif ragas_embeddings_provider in ['huggingface', 'sentence-transformers']:
            # Use HuggingFace embeddings (supports sentence-transformers models)
            embeddings = HuggingFaceEmbeddings(
                model_name=ragas_embeddings_model
            )
        else:
            raise ValueError(f"Unsupported RAGAS_EMBEDDINGS_PROVIDER: {ragas_embeddings_provider}")
        
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
        
        # Run evaluation with progress bar disabled
        results = evaluate(
            dataset,
            metrics=metric_objs,
            llm=llm,
            embeddings=embeddings,
            show_progress=False  # Disable progress bar for clean output
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
