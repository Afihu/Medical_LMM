"""
Gemini LLM Provider implementation.

This module provides the Gemini API integration following the LLMProvider interface.
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Union
import base64

from .base import LLMProvider, SetupError, GenerationError, ParseError


class GeminiProvider(LLMProvider):
    """LLM provider for Google's Gemini API."""
    
    def __init__(self):
        """Initialize the GeminiProvider."""
        self.model = None
        self.model_name = "models/gemini-2.5-flash"
        
    def setup(self, api_key: str = None, **kwargs) -> None:
        """
        Initialize and configure the Gemini API.
        
        Args:
            api_key: Gemini API key (optional, can be read from env)
            **kwargs: Additional config (model_name, etc.)
        
        Raises:
            SetupError: If setup fails
        """
        try:
            import google.generativeai as genai
            
            # Load configuration from config file and kwargs
            from scripts.config.llm_config import GEMINI_MODEL_NAME
            
            # Get API key from parameter or environment
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY")
                
            if not api_key:
                raise SetupError("Missing GEMINI_API_KEY in environment or parameter")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Set model name from config or kwargs
            self.model_name = kwargs.get("model_name", GEMINI_MODEL_NAME)
            
            # Create model instance
            self.model = genai.GenerativeModel(self.model_name)
            
        except ImportError:
            raise SetupError("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise SetupError(f"Failed to configure Gemini API: {e}")
    
    def generate_content(self, content: Union[List[Dict[str, Any]], Dict[str, Any]]) -> str:
        """
        Generate content using Gemini API.
        
        Args:
            content: Content in Gemini format:
                - List of content parts with text/inline_data
                - Or dict with text and images for conversion
        
        Returns:
            Generated text response
            
        Raises:
            GenerationError: If content generation fails
        """
        if not self.model:
            raise GenerationError("GeminiProvider not properly initialized. Call setup() first.")
        
        try:
            # Handle different content formats
            if isinstance(content, dict) and "text" in content:
                # Convert dict format to Gemini content parts
                content_parts = self._convert_dict_to_gemini_format(content)
            elif isinstance(content, list):
                # Already in Gemini format
                content_parts = content
            else:
                # Try to use content directly
                content_parts = content
            
            # Generate response
            response = self.model.generate_content(content_parts)
            return response.text.strip()
            
        except Exception as e:
            raise GenerationError(f"Gemini content generation failed: {e}")
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Gemini response as JSON.
        
        Handles cases where JSON is wrapped in Markdown code fences or has formatting issues.
        
        Args:
            response: Raw text response from Gemini
            
        Returns:
            Parsed JSON dict or error dict with raw_output
        """
        try:
            # Direct parse attempt
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try removing Markdown code fences
        try:
            # Check for ```json ... ``` or ``` ... ```
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
                    return json.loads(json_str)
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
                    return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try cleaning up common formatting issues
        try:
            # Remove extra whitespace and line breaks that might break JSON
            cleaned = response.strip()
            # Try to find JSON object start and end
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = cleaned[start_idx:end_idx+1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # If all parsing attempts fail, return dict with raw output
        return {
            "raw_output": response,
            "error": "Could not parse JSON - trying to extract from raw output",
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "analysis": "Failed to parse AI response",
            "differential_diagnosis": {},
            "confidence_level": "unknown"
        }
    
    def is_multimodal_supported(self) -> bool:
        """Gemini supports multimodal inputs (text + images)."""
        return True
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "gemini"
    
    def _convert_dict_to_gemini_format(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert dict format to Gemini content parts format.
        
        Args:
            content: Dict with "text" and optionally "images" (list of paths)
            
        Returns:
            List of Gemini content parts
        """
        content_parts = []
        
        # Add text part
        if "text" in content:
            content_parts.append({"text": content["text"]})
        
        # Add image parts
        if "images" in content:
            for img_path in content["images"]:
                try:
                    with open(img_path, "rb") as img_file:
                        img_data = img_file.read()
                        data = base64.b64encode(img_data).decode("utf-8")
                    
                    # Determine MIME type based on file extension
                    ext = os.path.splitext(img_path)[1].lower()
                    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
                    
                    content_parts.append({
                        "inline_data": {"mime_type": mime_type, "data": data}
                    })
                except Exception as e:
                    # Log warning but continue with other images
                    print(f"Warning: Could not process image {img_path}: {e}")
        
        return content_parts
    
    @staticmethod
    def extract_json(text: str) -> str:
        """Extract JSON from Markdown code blocks."""
        # Remove Markdown code block if present
        match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
        if match:
            return match.group(1).strip()
        return text.strip()