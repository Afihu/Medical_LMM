"""
LLM Services Module

This module provides a unified interface for interacting with different LLM providers
(Gemini, LM Studio, etc.) through a common abstraction layer.

Usage:
    from scripts.llm_services import get_llm_provider
    
    # Get provider based on environment variable LLM_PROVIDER
    provider = get_llm_provider()
    provider.setup(api_key="your-api-key")
    
    # Generate content
    response = provider.generate_content(content_parts)
"""

from .factory import get_llm_provider
from .base import LLMProvider

__all__ = ["get_llm_provider", "LLMProvider"]