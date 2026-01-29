"""
image_utils.py
--------------
Utility functions for compressing and resizing decoded images
before sending them to Gemini API or for display.
"""

import os
from PIL import Image

def compress_image(input_path, output_path=None, max_size=(512, 512), quality=80):
    """
    Compress and resize an image for Gemini input.

    Args:
        input_path (str): Path to the original image.
        output_path (str): Path to save the compressed version (default: same dir with '_compressed').
        max_size (tuple): Target maximum (width, height) in pixels.
        quality (int): JPEG quality (1–100).

    Returns:
        str: Path to the compressed image file.
    """
    try:
        img = Image.open(input_path)
        img.thumbnail(max_size, Image.LANCZOS)

        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_compressed.jpg"

        img.convert("RGB").save(output_path, "JPEG", quality=quality, optimize=True)

        return output_path

    except Exception as e:
        print(f"[compress_image] Error processing {input_path}: {e}")
        return input_path  # fallback to original
