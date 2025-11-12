"""
Abstract base class for LLM providers.

This module defines the interface that all LLM providers must implement
to ensure consistent behavior across different LLM backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    @abstractmethod
    def setup(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Initialize and configure the LLM provider.
        
        Args:
            api_key: API key for cloud providers (optional for local providers)
            **kwargs: Additional configuration parameters specific to each provider
            
        Raises:
            RuntimeError: If setup fails (connection issues, authentication, etc.)
        """
        pass
    
    @abstractmethod
    def generate_content(self, content: Union[List[Dict[str, Any]], Dict[str, Any], Any]) -> str:
        """
        Generate content from multimodal input.
        
        Args:
            content: Input content in provider-specific format:
                - For Gemini: List of content parts with text/inline_data
                - For LM Studio: Dict with text/images or lms.Chat object
                - For others: Provider-specific format
                
        Returns:
            Generated text response from the LLM
            
        Raises:
            RuntimeError: If content generation fails
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response as JSON (if applicable).
        
        Args:
            response: Raw text response from LLM
            
        Returns:
            Parsed JSON dict or error dict with raw_output and error message
        """
        pass
    
    def is_multimodal_supported(self) -> bool:
        """
        Check if the provider supports multimodal inputs (text + images).
        
        Returns:
            True if multimodal is supported, False otherwise
        """
        return False
    
    def get_provider_name(self) -> str:
        """
        Get the name of the provider for logging and debugging.
        
        Returns:
            Provider name (e.g., "gemini", "lmstudio", "ollama")
        """
        return self.__class__.__name__.lower().replace("provider", "")


class ProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class SetupError(ProviderError):
    """Exception raised when provider setup fails."""
    pass


class GenerationError(ProviderError):
    """Exception raised when content generation fails."""
    pass


class ParseError(ProviderError):
    """Exception raised when response parsing fails."""
    pass