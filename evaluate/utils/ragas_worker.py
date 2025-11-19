import sys
import json
import warnings
import os
import time
import re
from pathlib import Path

# Suppress all warnings for clean console output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress TensorFlow and related warnings BEFORE importing anything TensorFlow-related
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (0=all, 1=info, 2=warning, 3=error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations message
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Suppress GPU memory allocation messages

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
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from pydantic import Field, ConfigDict
from langchain_core.outputs import Generation
from evaluate.eval_config import RAGAS_EVALUATION_MODEL, RAGAS_EMBEDDINGS_MODEL

# Suppress absl logging after imports
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

# TensorFlow logging is already suppressed via environment variables above
try:
    import tensorflow
except ImportError:
    pass


def clean_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks.
    
    Handles patterns like:
    - ```json ... ```
    - ``` ... ```
    - Plain JSON
    """
    if not isinstance(text, str):
        return text
    
    text = text.strip()
    
    # Pattern 1: ```json ... ``` or ``` ... ```
    if text.startswith('```'):
        # Remove opening backticks and optional language specifier
        lines = text.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        # Remove closing backticks
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    
    return text


class RateLimitedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with configurable rate limiting."""
    
    # Configure Pydantic to allow extra fields
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
    
    # Declare custom fields
    api_request_delay: float = Field(default=0.0, description="Delay in seconds between API calls")
    last_request_time: float = Field(default=0.0, description="Timestamp of last API request")
    
    def __init__(self, *args, api_request_delay: float = 0.0, **kwargs):
        """Initialize with rate limiting.
        
        Args:
            api_request_delay: Delay in seconds between API calls
        """
        # Pass api_request_delay to parent now that it's declared
        super().__init__(*args, api_request_delay=api_request_delay, **kwargs)
        # Initialize last_request_time
        self.last_request_time = 0.0
    
    def _apply_rate_limit(self):
        """Apply rate limiting."""
        if self.api_request_delay > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.api_request_delay:
                time.sleep(self.api_request_delay - elapsed)
        self.last_request_time = time.time()
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> LLMResult:
        """Override _generate to add rate limiting."""
        self._apply_rate_limit()
        result = super()._generate(messages, stop, run_manager, **kwargs)
        
        # Clean JSON from markdown in responses by creating new Generation objects
        try:
            if result.generations:
                new_generations = []
                for generation_list in result.generations:
                    if not isinstance(generation_list, list):
                        new_generations.append(generation_list)
                        continue
                    new_gen_list = []
                    for generation in generation_list:
                        if hasattr(generation, 'text') and generation.text:
                            cleaned_text = clean_json_from_markdown(generation.text)
                            # Create new Generation object instead of modifying in place
                            new_gen = Generation(
                                text=cleaned_text,
                                generation_info=generation.generation_info if hasattr(generation, 'generation_info') else None
                            )
                            new_gen_list.append(new_gen)
                        else:
                            new_gen_list.append(generation)
                    new_generations.append(new_gen_list)
                # Create new LLMResult with cleaned generations
                result = LLMResult(
                    generations=new_generations,
                    llm_output=result.llm_output,
                    run=result.run if hasattr(result, 'run') else None
                )
        except Exception as e:
            # Log error but don't fail the generation
            import sys
            print(f"Warning: Error cleaning JSON from response: {e}", file=sys.stderr)
        
        return result
    
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager = None,
        **kwargs,
    ) -> LLMResult:
        """Override _agenerate to add rate limiting."""
        self._apply_rate_limit()
        result = await super()._agenerate(messages, stop, run_manager, **kwargs)
        
        # Clean JSON from markdown in responses by creating new Generation objects
        try:
            if result.generations:
                new_generations = []
                for generation_list in result.generations:
                    if not isinstance(generation_list, list):
                        new_generations.append(generation_list)
                        continue
                    new_gen_list = []
                    for generation in generation_list:
                        if hasattr(generation, 'text') and generation.text:
                            cleaned_text = clean_json_from_markdown(generation.text)
                            # Create new Generation object instead of modifying in place
                            new_gen = Generation(
                                text=cleaned_text,
                                generation_info=generation.generation_info if hasattr(generation, 'generation_info') else None
                            )
                            new_gen_list.append(new_gen)
                        else:
                            new_gen_list.append(generation)
                    new_generations.append(new_gen_list)
                # Create new LLMResult with cleaned generations
                result = LLMResult(
                    generations=new_generations,
                    llm_output=result.llm_output,
                    run=result.run if hasattr(result, 'run') else None
                )
        except Exception as e:
            # Log error but don't fail the generation
            import sys
            print(f"Warning: Error cleaning JSON from response: {e}", file=sys.stderr)
        
        return result


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
        
        # Get rate limiting configuration
        api_request_delay = float(input_data.get('api_request_delay', 0.0))
        
        # Local LLM configuration (for lmstudio/local providers)
        local_llm_url = input_data.get('local_llm_url', 'http://localhost:1234')
        lmstudio_model = input_data.get('lmstudio_model', 'medgemma-4b-it')
        lmstudio_temperature = input_data.get('lmstudio_temperature', 0.0)
        lmstudio_max_tokens = input_data.get('lmstudio_max_tokens', 32768)
        
        # Initialize LLM based on provider
        if ragas_llm_provider == 'gemini':
            if not api_key:
                raise ValueError("RAGAS_LLM_PROVIDER is 'gemini' but api_key not provided")
            # Use rate-limited Gemini class that also cleans JSON responses
            llm = RateLimitedChatGoogleGenerativeAI(
                model=RAGAS_EVALUATION_MODEL,
                google_api_key=api_key,
                temperature=0,
                api_request_delay=api_request_delay
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
        
        # Configure RunConfig for strict rate limiting
        # max_workers=1 ensures only 1 concurrent LLM call, preventing race conditions
        # This allows our custom RateLimitedChatGoogleGenerativeAI delays to work correctly
        run_config = RunConfig(
            max_workers=1,      # Sequential execution - critical for rate limiting
            timeout=180,        # 3 minute timeout per LLM call
            max_retries=3,      # Retry on transient failures
            max_wait=60         # Max wait between retries
        )
        
        # Run evaluation with progress bar disabled
        results = evaluate(
            dataset,
            metrics=metric_objs,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,  # Apply concurrency control
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
