"""
Test script for local LLM evaluation support (Phase 2 & 3)
----------------------------------------------------------
Tests RAGAS evaluation with different LLM and embeddings providers.

Usage:
    # Test with Gemini (default)
    python evaluate/test_local_llm_eval.py --provider gemini
    
    # Test with local LLM (LM Studio)
    python evaluate/test_local_llm_eval.py --provider local
    
    # Test both providers
    python evaluate/test_local_llm_eval.py --provider both
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluate.utils.ragas_evaluator import RAGASEvaluator
from scripts.llm_services.factory import setup_provider_from_env


def create_test_samples():
    """Create minimal test samples for RAGAS evaluation."""
    return [
        {
            'user_input': 'Patient presents with fever, headache, and muscle pain after recent travel to Southeast Asia.',
            'retrieved_contexts': [
                'Dengue Fever: Common in Southeast Asia. Symptoms include high fever, severe headache, pain behind the eyes, muscle and joint pain.',
                'Malaria: Endemic in tropical regions. Presents with fever, chills, and flu-like symptoms.',
                'Typhoid Fever: Waterborne disease common in developing countries. Causes sustained fever, weakness, and abdominal pain.'
            ],
            'response': 'Dengue Fever (high likelihood) - The patient\'s symptoms and recent travel to Southeast Asia strongly suggest dengue fever.',
            'reference': 'Dengue Fever'
        }
    ]


def test_ragas_with_provider(provider_name: str):
    """Test RAGAS evaluation with specified provider.
    
    Args:
        provider_name: 'gemini' or 'local'
    """
    print(f"\n{'='*70}")
    print(f"Testing RAGAS Evaluation with {provider_name.upper()} Provider")
    print(f"{'='*70}\n")
    
    # Set environment variables for this test
    original_llm_provider = os.environ.get('LLM_PROVIDER')
    original_ragas_llm = os.environ.get('RAGAS_LLM_PROVIDER')
    original_ragas_emb = os.environ.get('RAGAS_EMBEDDINGS_PROVIDER')
    
    try:
        # Configure providers
        if provider_name == 'gemini':
            os.environ['LLM_PROVIDER'] = 'gemini'
            os.environ['RAGAS_LLM_PROVIDER'] = 'gemini'
            os.environ['RAGAS_EMBEDDINGS_PROVIDER'] = 'google'
            
            # Check API key
            if not os.getenv('GEMINI_API_KEY'):
                print("❌ GEMINI_API_KEY not found in environment")
                print("   Please set GEMINI_API_KEY in .env file")
                return False
                
        elif provider_name == 'local':
            os.environ['LLM_PROVIDER'] = 'lmstudio'
            os.environ['RAGAS_LLM_PROVIDER'] = 'local'
            os.environ['RAGAS_EMBEDDINGS_PROVIDER'] = 'huggingface'
            
            # Check local LLM URL
            local_url = os.getenv('LOCAL_LLM_URL', 'http://localhost:1234')
            print(f"📡 Using local LLM at: {local_url}")
            print(f"   Make sure LM Studio or compatible server is running!")
        
        # Initialize LLM provider
        print(f"\n1️⃣  Initializing LLM provider...")
        llm_provider = setup_provider_from_env()
        print(f"   ✅ LLM Provider: {llm_provider.get_provider_name()}")
        
        # Initialize RAGAS evaluator
        print(f"\n2️⃣  Initializing RAGAS evaluator...")
        evaluator = RAGASEvaluator(llm_provider)
        
        # Create test samples
        print(f"\n3️⃣  Creating test samples...")
        samples = create_test_samples()
        print(f"   ✅ Created {len(samples)} test sample(s)")
        
        # Run RAGAS evaluation
        print(f"\n4️⃣  Running RAGAS evaluation...")
        print(f"   Metrics: context_precision, context_recall, faithfulness")
        
        scores = evaluator._evaluate_in_subprocess(
            samples=samples,
            metrics=['context_precision', 'context_recall', 'faithfulness']
        )
        
        # Display results
        print(f"\n5️⃣  Evaluation Results:")
        if scores:
            print(f"   ✅ SUCCESS!")
            for metric, value in scores.items():
                if value is not None:
                    print(f"      {metric}: {value:.4f}")
                else:
                    print(f"      {metric}: None (failed)")
            
            # Check if all metrics succeeded
            all_succeeded = all(v is not None for v in scores.values())
            if all_succeeded:
                print(f"\n   🎉 All metrics computed successfully!")
                return True
            else:
                print(f"\n   ⚠️  Some metrics failed to compute")
                return False
        else:
            print(f"   ❌ FAILED - No scores returned")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original environment variables
        if original_llm_provider:
            os.environ['LLM_PROVIDER'] = original_llm_provider
        elif 'LLM_PROVIDER' in os.environ:
            del os.environ['LLM_PROVIDER']
            
        if original_ragas_llm:
            os.environ['RAGAS_LLM_PROVIDER'] = original_ragas_llm
        elif 'RAGAS_LLM_PROVIDER' in os.environ:
            del os.environ['RAGAS_LLM_PROVIDER']
            
        if original_ragas_emb:
            os.environ['RAGAS_EMBEDDINGS_PROVIDER'] = original_ragas_emb
        elif 'RAGAS_EMBEDDINGS_PROVIDER' in os.environ:
            del os.environ['RAGAS_EMBEDDINGS_PROVIDER']


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description='Test RAGAS evaluation with different providers')
    parser.add_argument(
        '--provider',
        choices=['gemini', 'local', 'both'],
        default='gemini',
        help='Which provider to test (default: gemini)'
    )
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"RAGAS Local LLM Evaluation Test Suite")
    print(f"Phase 2 & 3 Implementation Validation")
    print(f"{'='*70}\n")
    
    results = {}
    
    if args.provider in ['gemini', 'both']:
        results['gemini'] = test_ragas_with_provider('gemini')
    
    if args.provider in ['local', 'both']:
        results['local'] = test_ragas_with_provider('local')
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}\n")
    
    for provider, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {provider.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n🎉 All tests PASSED! Phase 2 & 3 implementation is working correctly.\n")
        return 0
    else:
        print(f"\n⚠️  Some tests FAILED. Please check the output above for details.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
