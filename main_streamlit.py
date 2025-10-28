"""
main_streamlit.py
-----------------
Streamlit interface for the Medical_LMM system.
Users can enter text input like a chat, and see Gemini’s output in real time.
Each conversation is also logged to /responses/ as a Markdown file for debugging.
"""

import os
from datetime import datetime
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
#import io
import numpy as np

# Custom modules
from scripts.query import run_query
from scripts.prompt_generate import generate_prompt
from scripts.streamlit_ui_helper import load_conversation, save_conversation, save_uploaded_image
from scripts.u_i_a import embed_img as embed_image
from scripts.clean_folder import clean_folder


# --- Path configuration ---
# Get the project root (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CASES_FOLDER_TEXT = os.path.join(PROJECT_ROOT, "cases_text")
CASES_FOLDER_IMAGE = os.path.join(PROJECT_ROOT, "cases_image")
RESPONSES_FOLDER = os.path.join(PROJECT_ROOT, "responses")
UPLOADED_IMAGES_FOLDER = os.path.join(PROJECT_ROOT, "uploaded_images")
DECODED_IMG_FOLDER = os.path.join(PROJECT_ROOT, "decoded_images")

# --- Configuration ---
MODEL_NAME = "models/gemini-2.5-pro"

# --- Setup ---
def setup():
    """Configure Gemini API and ensure directories exist."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY in .env file.")
        st.stop()

    genai.configure(api_key=api_key)
    os.makedirs(RESPONSES_FOLDER, exist_ok=True)
    os.makedirs(CASES_FOLDER_TEXT, exist_ok=True)
    os.makedirs(CASES_FOLDER_IMAGE, exist_ok=True)
    os.makedirs(UPLOADED_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(DECODED_IMG_FOLDER, exist_ok=True)

# --- Main Loop ---
def main():
    setup()
    st.set_page_config(page_title="Medical LMM", page_icon="💬", layout="wide")
    st.title("💬 Medical LMM Diagnostic Assistant")
    st.caption("AI-assisted medical case reasoning using Gemini + Qdrant")

    # --- Sidebar: history ---
    with st.sidebar:
        st.header("🗂️ Chat History")
        response_files = sorted(
            [f for f in os.listdir(RESPONSES_FOLDER) if f.endswith(".md")],
            reverse=True
        )
        selected_file = st.selectbox("Select a past conversation:", ["(New Chat)"] + response_files)

        if selected_file != "(New Chat)":
            st.session_state["messages"] = load_conversation(os.path.join(RESPONSES_FOLDER, selected_file))
            st.info(f"Loaded conversation from: {selected_file}")
        else:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

    # --- Display chat ---
    for msg in st.session_state["messages"]:
        if msg["role"] == "user" and msg.get("image"):  # display image that user has added for querying
            with st.chat_message("user"):
                st.write(msg["content"])
                st.image(msg["image"], caption="User image", use_container_width=True)
        else:
            st.chat_message(msg["role"]).write(msg["content"])

    # --- Chat input ---
    st.divider()
    col1, col2 = st.columns([8, 1])
    with col1:
        prompt = st.text_input(
            "Enter patient's symptoms and history...",
            key="chat_input",
            label_visibility="collapsed",
        )
    with col2:
        uploaded_files = st.file_uploader(
            "➕",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="image_input",
            accept_multiple_files=True,  # allow multiple images
        )

    # --- Submit button ---
    if st.button("Send", use_container_width=True):
        if not prompt and not uploaded_files:
            st.warning("Please enter text or upload an image.")
            st.stop()

        model = genai.GenerativeModel(MODEL_NAME)

        # --- Clear old uploads & save new images ---
        image_paths = []
        if uploaded_files:
            # Clean all relevant temp folders before new query
            print("\nClearing temporary directories for a new query...\n")
            clean_folder(os.path.join(PROJECT_ROOT, "uploaded_images"))
            clean_folder(os.path.join(PROJECT_ROOT, "decoded_images"))
            clean_folder(os.path.join(PROJECT_ROOT, "cases_image"))
            clean_folder(os.path.join(PROJECT_ROOT, "cases_text"))
            print("\nClearing done! Begin saving uplaoded images...\n")
            for f in uploaded_files:
                path = save_uploaded_image(f)
                image_paths.append(path)


        # Add user message to chat
        user_msg = {"role": "user", "content": prompt or ""}
        if image_paths:  # already contains saved file paths
            user_msg["images"] = image_paths
        st.session_state["messages"].append(user_msg)

        # --- Display in chat ---
        with st.chat_message("user"):
            if prompt:
                st.write(prompt)
            if image_paths:
                for img in image_paths:
                    st.image(img, caption="Uploaded image", use_container_width=True)

        # --- Embedding placeholders ---
        text_vector = None
        image_vector = None

        if prompt:
            # TODO: Replace with actual text embedding
            text_vector = [0.12, -0.45, 0.78, 0.66]

        if uploaded_files:
            uploaded_folder = os.path.join(PROJECT_ROOT, "uploaded_images")

            # Expect exactly one image file
            image_files = [
                os.path.join(uploaded_folder, f)
                for f in os.listdir(uploaded_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            if not image_files:
                st.warning("No images found in /uploaded_images to embed.")
            else:
                image_path = image_files[0]
                st.info(f"Embedding image: {os.path.basename(image_path)}")

                try:
                    # Use filename (without extension) as caption
                    caption = os.path.splitext(os.path.basename(image_path))[0]
                    emb = embed_image(image_path, caption=[caption])

                    # Extract actual image embedding tensor
                    if hasattr(emb, "image_embeds"):
                        image_vector = emb.image_embeds.squeeze().tolist()
                    else:
                        st.error("Embedding output missing 'image_embeds' attribute.")
                        image_vector = None

                    if image_vector:
                        st.success(f"Computed image embedding of dimension {len(image_vector)}")
                    else:
                        st.warning("Image embedding returned empty vector.")

                except Exception as e:
                    st.error(f"Error embedding {os.path.basename(image_path)}: {e}")
                    image_vector = None
        else:
            image_vector = None


        # # Step 1: Query Qdrant
        if text_vector:
            run_query(text_vector, mode="text")
        if image_vector:
            run_query(image_vector, mode="image")

        # Step 2: Build prompt
        with st.spinner("Building prompt and generating analysis..."):
            # generate_prompt now returns (text_prompt, decoded_image_paths)
            final_prompt, decoded_images_path = generate_prompt(prompt)

            # Build multimodal content parts: text followed by inline images
            content_parts = [{"text": final_prompt}]

            # Attach decoded images if any
            for img_path in decoded_images_path:
                try:
                    with open(img_path, "rb") as f:
                        data = f.read()
                    content_parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": data
                        }
                    })
                except Exception as e:
                    st.warning(f"Could not attach image {img_path}: {e}")
            
            # Send to Gemini as multimodal parts
            try:
                response = model.generate_content(content_parts)
                ai_text = response.text
            except Exception as e:
                st.error(f"Error calling Gemini: {e}")
                ai_text = f"Error during model generation: {e}"

        # Step 3: Display AI response
        st.session_state["messages"].append({"role": "assistant", "content": ai_text})
        st.chat_message("assistant").write(ai_text)

        # Step 4: Save full chat to markdown
        filename = save_conversation(st.session_state["messages"])
        st.sidebar.success(f"Conversation saved as {filename}")

# --- Run app ---
if __name__ == "__main__":
    main()
