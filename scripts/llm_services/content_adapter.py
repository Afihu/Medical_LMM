"""
Content adapter for converting between different LLM provider formats.

This module provides utilities to convert content between the different
formats expected by various LLM providers (Gemini, LM Studio, etc.).
"""

import base64
import os
from typing import List, Dict, Any, Union, Optional


class ContentAdapter:
    """Adapter class for converting content between LLM provider formats."""
    
    @staticmethod
    def to_gemini_format(text: str, image_paths: Optional[List[str]] = None, 
                        system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert text and images to Gemini content parts format.
        
        Args:
            text: Text content
            image_paths: List of image file paths (optional)
            system_prompt: System prompt (note: Gemini doesn't use system prompts in content)
            
        Returns:
            List of Gemini content parts with text and inline_data for images
        """
        content_parts = []
        
        # Add text part (combine system prompt with user text if needed)
        full_text = text
        if system_prompt:
            full_text = f"{system_prompt}\n\n{text}"
        
        content_parts.append({"text": full_text})
        
        # Add image parts
        if image_paths:
            for img_path in image_paths:
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
                    print(f"Warning: Could not process image {img_path}: {e}")
        
        return content_parts
    
    @staticmethod
    def to_lmstudio_format(text: str, image_paths: Optional[List[str]] = None,
                          system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert text and images to LM Studio format (dict that will be converted to Chat object).
        
        Args:
            text: Text content
            image_paths: List of image file paths (optional)
            system_prompt: System prompt for Chat initialization
            
        Returns:
            Dict with "text", "images", and optionally "system" keys
        """
        result = {"text": text}
        
        if system_prompt:
            result["system"] = system_prompt
        
        if image_paths:
            result["images"] = image_paths
        
        return result
    
    @staticmethod
    def from_gemini_format(content_parts: List[Dict[str, Any]]) -> tuple[str, List[str], Optional[str]]:
        """
        Extract text and images from Gemini content parts format.
        
        Args:
            content_parts: List of Gemini content parts
            
        Returns:
            Tuple of (text, image_paths, system_prompt)
            Note: image_paths will be empty as Gemini uses inline base64 data
        """
        text_parts = []
        image_count = 0
        
        for part in content_parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "inline_data" in part:
                # We can't easily convert back to file paths from base64
                # Just note that there was an image
                image_count += 1
        
        text = " ".join(text_parts)
        
        # Return empty image paths since we can't reconstruct files from base64
        return text, [], None
    
    @staticmethod
    def normalize_to_dict(content: Union[List[Dict], Dict, str], 
                         default_system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Normalize various content formats to a standard dict format.
        
        Args:
            content: Content in various formats
            default_system_prompt: Default system prompt to use if none provided
            
        Returns:
            Dict with "text", "images" (if any), and "system" (if any) keys
        """
        if isinstance(content, str):
            return {
                "text": content,
                "system": default_system_prompt
            }
        
        elif isinstance(content, dict):
            # Already in dict format, ensure it has required keys
            result = content.copy()
            if "text" not in result:
                result["text"] = ""
            if default_system_prompt and "system" not in result:
                result["system"] = default_system_prompt
            return result
        
        elif isinstance(content, list):
            # Assume Gemini-style content parts
            text, image_paths, system = ContentAdapter.from_gemini_format(content)
            result = {"text": text}
            
            if image_paths:
                result["images"] = image_paths
            
            if system or default_system_prompt:
                result["system"] = system or default_system_prompt
                
            return result
        
        else:
            # Try to convert to string
            return {
                "text": str(content),
                "system": default_system_prompt
            }
    
    @staticmethod
    def add_medical_system_prompt(content: Union[Dict, List, str], 
                                 medical_prompt: Optional[str] = None) -> Union[Dict, List]:
        """
        Add medical-specific system prompt to content.
        
        Args:
            content: Content in various formats
            medical_prompt: Medical system prompt to add (uses default if None)
            
        Returns:
            Content with medical system prompt added
        """
        if medical_prompt is None:
            medical_prompt = (
                "You are a medical AI assistant. Provide accurate, evidence-based medical information. "
                "Always include differential diagnosis considerations and recommend consulting healthcare professionals. "
                "Format your response as JSON with 'analysis' and 'differential_diagnosis' fields."
            )
        
        if isinstance(content, dict):
            result = content.copy()
            result["system"] = medical_prompt
            return result
        
        elif isinstance(content, list):
            # For Gemini format, prepend system prompt to first text part
            if content and "text" in content[0]:
                content[0]["text"] = f"{medical_prompt}\n\n{content[0]['text']}"
            else:
                content.insert(0, {"text": medical_prompt})
            return content
        
        elif isinstance(content, str):
            return {"text": content, "system": medical_prompt}
        
        else:
            return {"text": str(content), "system": medical_prompt}
    
    @staticmethod
    def validate_image_paths(image_paths: List[str]) -> List[str]:
        """
        Validate and filter image paths, removing non-existent or invalid files.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of valid image file paths
        """
        valid_paths = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for path in image_paths:
            try:
                if os.path.exists(path):
                    ext = os.path.splitext(path)[1].lower()
                    if ext in supported_extensions:
                        valid_paths.append(path)
                    else:
                        print(f"Warning: Unsupported image format {ext} for {path}")
                else:
                    print(f"Warning: Image file not found: {path}")
            except Exception as e:
                print(f"Warning: Error validating image path {path}: {e}")
        
        return valid_paths


# Convenience functions for common conversions
def to_provider_format(content: Union[Dict, List, str], provider: str, 
                      system_prompt: Optional[str] = None) -> Union[Dict, List]:
    """
    Convert content to the format expected by a specific provider.
    
    Args:
        content: Content in various formats
        provider: Target provider ("gemini" or "lmstudio")
        system_prompt: Optional system prompt
        
    Returns:
        Content in provider-specific format
    """
    # Normalize to dict first
    normalized = ContentAdapter.normalize_to_dict(content, system_prompt)
    
    if provider.lower() == "gemini":
        return ContentAdapter.to_gemini_format(
            text=normalized.get("text", ""),
            image_paths=normalized.get("images"),
            system_prompt=normalized.get("system")
        )
    elif provider.lower() in ["lmstudio", "lm_studio"]:
        return ContentAdapter.to_lmstudio_format(
            text=normalized.get("text", ""),
            image_paths=normalized.get("images"),
            system_prompt=normalized.get("system")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def add_medical_context(content: Union[Dict, List, str]) -> Union[Dict, List]:
    """
    Add medical AI context to content.
    
    Args:
        content: Content in various formats
        
    Returns:
        Content with medical system prompt added
    """
    return ContentAdapter.add_medical_system_prompt(content)