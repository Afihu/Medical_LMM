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
from PIL import Image
#import io
import numpy as np

# Custom modules
from scripts.query import run_query
from scripts.prompt_generate import generate_prompt
from scripts.streamlit_ui_helper import load_conversation, save_conversation, save_uploaded_image
from scripts.u_i_a import embed_img as embed_image

# --- Path configuration ---
# Get the project root (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CASES_FOLDER_TEXT = os.path.join(PROJECT_ROOT, "cases_text")
CASES_FOLDER_IMAGE = os.path.join(PROJECT_ROOT, "cases_image")
RESPONSES_FOLDER = os.path.join(PROJECT_ROOT, "responses")
UPLOADED_IMAGES_FOLDER = os.path.join(PROJECT_ROOT, "uploaded_images")

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
    # if prompt := st.chat_input("Enter new patient's symptoms and history..."):
    #     model = genai.GenerativeModel(MODEL_NAME)
    #     st.session_state["messages"].append({"role": "user", "content": prompt})
    #     st.chat_message("user").write(prompt)
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

        # --- Save uploaded images ---
        image_paths = []
        if uploaded_files:
            for f in uploaded_files:
                path = save_uploaded_image(f)
                image_paths.append(path)

        # Add user message to chat
        user_msg = {"role": "user", "content": prompt or ""}
        if uploaded_files:
            image = Image.open(uploaded_files)
            user_msg["image"] = image
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

            # Collect all image files in /uploaded_images
            image_files = [
                os.path.join(uploaded_folder, f)
                for f in os.listdir(uploaded_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            if not image_files:
                st.warning("No images found in /uploaded_images to embed.")
            else:
                st.info(f"Found {len(image_files)} uploaded image(s). Computing embeddings...")

                image_embeddings = []
                for image_path in image_files:
                    try:
                        emb = embed_image(image_path, caption=None)
                        image_embeddings.append(emb[0])  # emb[0] = single vector for one image
                    except Exception as e:
                        st.error(f"Error embedding {os.path.basename(image_path)}: {e}")

                # Average embeddings across all images
                if image_embeddings:
                    image_vector = np.mean(np.array(image_embeddings), axis=0).tolist()
                    st.success(f"Computed averaged image embedding of dimension {len(image_vector)}")
                else:
                    image_vector = None


        # # Step 1: Query Qdrant
        if text_vector:
            run_query(text_vector, mode="text")
        if image_vector:
            run_query(image_vector, mode="image")

        # Step 2: Build prompt
        with st.spinner("Building prompt and generating analysis..."):
            final_prompt = generate_prompt(prompt)
            response = model.generate_content(final_prompt)
            ai_text = response.text

        # Step 3: Display AI response
        st.session_state["messages"].append({"role": "assistant", "content": ai_text})
        st.chat_message("assistant").write(ai_text)

        # Step 4: Save full chat to markdown
        filename = save_conversation(st.session_state["messages"])
        st.sidebar.success(f"Conversation saved as {filename}")

# --- Run app ---
if __name__ == "__main__":
    main()
