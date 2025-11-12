"""
Factory module for creating LLM providers.

This module provides the factory pattern for instantiating the appropriate
LLM provider based on configuration or runtime parameters.
"""

import os
from typing import Optional

from .base import LLMProvider, SetupError
from .gemini_provider import GeminiProvider
from .lmstudio_provider import LMStudioProvider


def get_llm_provider(provider_type: Optional[str] = None) -> LLMProvider:
    """
    Get LLM provider based on configuration or parameter.
    
    Args:
        provider_type: Provider type ("gemini", "lmstudio", or None for auto-detection)
                      If None, uses LLM_PROVIDER environment variable, defaults to "gemini"
        
    Returns:
        Configured LLMProvider instance
        
    Raises:
        ValueError: If unknown provider type is specified
        SetupError: If provider cannot be instantiated
        
    Examples:
        # Use environment variable LLM_PROVIDER
        provider = get_llm_provider()
        
        # Explicitly specify provider
        provider = get_llm_provider("gemini")
        provider = get_llm_provider("lmstudio")
    """
    # Determine provider type
    if provider_type is None:
        provider_type = os.getenv("LLM_PROVIDER", "gemini").lower()
    else:
        provider_type = provider_type.lower()
    
    # Create provider based on type
    if provider_type == "gemini":
        return GeminiProvider()
    elif provider_type in ["lmstudio", "lm_studio", "lmstudio_provider"]:
        return LMStudioProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_type}. "
                        f"Supported providers: gemini, lmstudio")


def get_available_providers() -> list[str]:
    """
    Get list of available provider types.
    
    Returns:
        List of provider type strings
    """
    return ["gemini", "lmstudio"]


def setup_provider_from_env() -> LLMProvider:
    """
    Create and setup LLM provider using configuration file and environment variables.
    
    Uses config file for provider selection and settings, environment for API keys:
    - Config file: Provider type, model names, parameters
    - Environment: GEMINI_API_KEY for Gemini provider
    
    Returns:
        Configured and ready-to-use LLMProvider instance
        
    Raises:
        SetupError: If provider setup fails
        ValueError: If unknown provider is specified
        
    Examples:
        # Set environment variables first:
        # export GEMINI_API_KEY=your_api_key
        
        provider = setup_provider_from_env()
        response = provider.generate_content([{"text": "Hello"}])
    """
    # Load configuration from config file
    from scripts.config.llm_config import LLM_PROVIDER
    
    provider = get_llm_provider(LLM_PROVIDER)
    
    # Setup provider with config file and environment-based configuration
    try:
        if provider.get_provider_name() == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            provider.setup(api_key=api_key)
            
        elif provider.get_provider_name() == "lmstudio":
            provider.setup()  # Uses config file settings
            
        return provider
        
    except Exception as e:
        raise SetupError(f"Failed to setup {provider.get_provider_name()} provider: {e}")


def validate_provider_config(provider_type: str = None) -> tuple[bool, str]:
    """
    Validate if the configuration for a provider type is available.
    
    Args:
        provider_type: Provider type to validate (if None, reads from config file)
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        is_valid, error = validate_provider_config("gemini")
        if not is_valid:
            print(f"Configuration error: {error}")
    """
    # If no provider type specified, read from config file
    if provider_type is None:
        try:
            from scripts.config.llm_config import LLM_PROVIDER
            provider_type = LLM_PROVIDER
        except ImportError:
            return False, "Cannot import LLM configuration from scripts.config.llm_config"
    
    provider_type = provider_type.lower()
    
    if provider_type == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return False, "GEMINI_API_KEY environment variable is not set"
        return True, ""
        
    elif provider_type in ["lmstudio", "lm_studio"]:
        # Load config from config file
        try:
            from scripts.config.llm_config import LMSTUDIO_URL
            
            # Basic validation - check if URL format is reasonable
            if not LMSTUDIO_URL.startswith(("http://", "https://")):
                return False, f"Invalid LMSTUDIO_URL format in config: {LMSTUDIO_URL}"
            
            # Note: We don't test actual connectivity here as it would require
            # making a network call
            return True, ""
            
        except ImportError:
            return False, "Cannot import LM Studio configuration from scripts.config.llm_config"
        
    else:
        return False, f"Unknown provider type: {provider_type}"