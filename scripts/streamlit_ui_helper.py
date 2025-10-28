"""
streamlit_ui_helper.py
----------------------
Helper functions for Streamlit interface:
- load_conversation(): read a markdown chat history file
- save_conversation(): write current chat messages to markdown
"""

import os
from datetime import datetime
from PIL import Image

# --- Path configuration ---
# Get the project root (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # as this will return /scripts, so we wrap in another one
RESPONSES_FOLDER = os.path.join(PROJECT_ROOT, "responses")
UPLOADED_IMAGES_FOLDER = os.path.join(PROJECT_ROOT, "uploaded_images")

def load_conversation(md_path):
    """Parse a Markdown conversation file into a list of messages."""
    messages = []
    if not os.path.exists(md_path):
        return messages

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("**User:**"):
            messages.append({"role": "user", "content": line.replace("**User:**", "").strip()})
        elif line.startswith("**AI:**"):
            messages.append({"role": "assistant", "content": line.replace("**AI:**", "").strip()})
    return messages


def save_conversation(messages):
    """Save a chat conversation to /responses as a Markdown file."""
    os.makedirs(RESPONSES_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{timestamp}.md"
    filepath = os.path.join(RESPONSES_FOLDER, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for msg in messages:
            role = "User" if msg["role"] == "user" else "AI"
            f.write(f"**{role}:** {msg['content']}\n\n")

    return filename

def save_uploaded_image(uploaded_file):
    """Save uploaded image to /uploaded_images with unique filename and return its path."""
    os.makedirs(UPLOADED_IMAGES_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"user_image_{timestamp}.png"
    save_path = os.path.join(UPLOADED_IMAGES_FOLDER, filename)
    image = Image.open(uploaded_file)
    image.save(save_path)
    return save_path