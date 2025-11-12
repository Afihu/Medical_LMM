"""
LM Studio LLM Provider implementation.

This module provides the LM Studio (OpenAI-compatible API) integration following the LLMProvider interface.
LM Studio uses an OpenAI-compatible API at /v1/chat/completions endpoint.
"""

import json
import os
import base64
from datetime import datetime
from typing import List, Dict, Any, Union, Optional

from .base import LLMProvider, SetupError, GenerationError, ParseError


class LMStudioProvider(LLMProvider):
    """LLM provider for LM Studio using OpenAI-compatible API."""
    
    def __init__(self):
        """Initialize the LMStudioProvider."""
        self.model_name = None
        self.url = None
        self.temperature = 0.7
        self.max_tokens = 1000
        self.top_p = 0.95
        
    def setup(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Initialize and configure the LM Studio connection.
        
        Args:
            api_key: Not used for LM Studio (local inference)
            **kwargs: Additional config
        
        Raises:
            SetupError: If setup fails
        """
        try:
            import requests
        except ImportError:
            raise SetupError("requests not installed. Run: pip install requests")
        
        from scripts.config.llm_config import (
            LMSTUDIO_MODEL, LMSTUDIO_URL, LMSTUDIO_TEMPERATURE, 
            LMSTUDIO_MAX_TOKENS, LMSTUDIO_TOP_P
        )
        
        self.model_name = kwargs.get("model_name", LMSTUDIO_MODEL)
        self.url = kwargs.get("url", LMSTUDIO_URL)
        self.temperature = float(kwargs.get("temperature", LMSTUDIO_TEMPERATURE))
        self.max_tokens = int(kwargs.get("max_tokens", LMSTUDIO_MAX_TOKENS))
        self.top_p = float(kwargs.get("top_p", LMSTUDIO_TOP_P))
        
        try:
            health_url = f"{self.url.rstrip('/')}/v1/models"
            response = requests.get(health_url, timeout=5)
            if response.status_code != 200:
                raise SetupError(f"LM Studio server returned {response.status_code}")
        except requests.RequestException as e:
            raise SetupError(f"Cannot connect to LM Studio at {self.url}: {e}")
    
    def generate_content(self, content: Union[List[Dict[str, Any]], Dict[str, Any], Any]) -> str:
        """
        Generate content using LM Studio OpenAI-compatible API.
        
        Args:
            content: Content in various formats
        
        Returns:
            Generated text response
            
        Raises:
            GenerationError: If content generation fails
        """
        try:
            import requests
            
            messages = self._convert_to_chat_messages(content)
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "stream": False
            }
            
            url = f"{self.url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                raise GenerationError(f"LM Studio API error {response.status_code}: {response.text}")
            
            result = response.json()
            if "choices" not in result or len(result["choices"]) == 0:
                raise GenerationError("No response choices returned from LM Studio")
            
            return result["choices"][0]["message"]["content"]
            
        except requests.RequestException as e:
            raise GenerationError(f"LM Studio request failed: {e}")
        except Exception as e:
            raise GenerationError(f"LM Studio content generation failed: {e}")
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LM Studio response as JSON.
        
        Args:
            response: Raw text response from LM Studio
            
        Returns:
            Parsed JSON dict or error dict with raw_output
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                cleaned = response.strip()
                if "```json" in cleaned:
                    start = cleaned.find("```json") + 7
                    end = cleaned.find("```", start)
                    if end != -1:
                        json_str = cleaned[start:end].strip()
                        return json.loads(json_str)
                elif "```" in cleaned:
                    start = cleaned.find("```") + 3
                    end = cleaned.find("```", start)
                    if end != -1:
                        json_str = cleaned[start:end].strip()
                        return json.loads(json_str)
                
                start_idx = cleaned.find('{')
                end_idx = cleaned.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = cleaned[start_idx:end_idx+1]
                    return json.loads(json_str)
                    
            except json.JSONDecodeError:
                pass
        
        return {
            "raw_output": response,
            "error": "Could not parse JSON response",
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "analysis": "Failed to parse AI response",
            "differential_diagnosis": {},
            "confidence_level": "unknown"
        }
    
    def is_multimodal_supported(self) -> bool:
        """LM Studio supports multimodal if using a VLM (Vision-Language Model)."""
        return True
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "lmstudio"
    
    def _convert_to_chat_messages(self, content: Any) -> List[Dict[str, Any]]:
        """
        Convert various content formats to OpenAI chat messages format.
        
        Args:
            content: Content in various formats
            
        Returns:
            List of chat messages in OpenAI format
        """
        messages = []
        
        if isinstance(content, dict):
            if "system" in content:
                messages.append({
                    "role": "system",
                    "content": content["system"]
                })
            
            user_content = []
            
            if "text" in content:
                user_content.append({
                    "type": "text",
                    "text": content["text"]
                })
            
            if "images" in content and content["images"]:
                for img_path in content["images"]:
                    try:
                        with open(img_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                        
                        ext = os.path.splitext(img_path)[1].lower()
                        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
                        
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_data}"
                            }
                        })
                    except Exception as e:
                        print(f"Warning: Could not process image {img_path}: {e}")
            
            if user_content:
                if len(user_content) == 1 and user_content[0]["type"] == "text":
                    messages.append({
                        "role": "user",
                        "content": user_content[0]["text"]
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": user_content
                    })
        
        elif isinstance(content, list):
            user_parts = []
            
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    user_parts.append(part["text"])
            
            if user_parts:
                messages.append({
                    "role": "user",
                    "content": " ".join(user_parts)
                })
        
        elif isinstance(content, str):
            messages.append({
                "role": "user",
                "content": content
            })
        
        else:
            messages.append({
                "role": "user",
                "content": str(content)
            })
        
        return messages
