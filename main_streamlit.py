"""
main_streamlit.py (Refactored)
------------------------------
Streamlit interface for the Medical_LMM system (Refactored Main Pipeline).
Uses QueryOrchestrator to handle user query embeddings for both text and image.
Temporary data is staged in /temp_query_data/<session_id>/ for each session.
"""

import os
from datetime import datetime
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import json

# Custom modules
from scripts.qdrant_services.query import run_query
from scripts.main_runtime.prompt_generate import generate_prompt
from scripts.main_runtime.streamlit_ui_helper import load_conversation, save_conversation, save_uploaded_image
from scripts.main_runtime.clean_folder import clean_folder
from scripts.embedding_generation_module.orchestrators import QueryOrchestrator

# --- Path configuration ---
# Get the project root (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DIAGNOSED_CASES_FOLDER = os.path.join(PROJECT_ROOT, "diagnosed_cases")
UPLOADED_IMAGES_FOLDER = os.path.join(PROJECT_ROOT, "uploaded_images")
TEMP_QUERY_DATA = os.path.join(PROJECT_ROOT, "temp_query_data")

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
    os.makedirs(DIAGNOSED_CASES_FOLDER, exist_ok=True)
    os.makedirs(UPLOADED_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(TEMP_QUERY_DATA, exist_ok=True)

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
            [f for f in os.listdir(DIAGNOSED_CASES_FOLDER) if f.endswith(".md")],
            reverse=True
        )
        selected_file = st.selectbox("Select a past conversation:", ["(New Chat)"] + response_files)

        if selected_file != "(New Chat)":
            st.session_state["messages"] = load_conversation(os.path.join(DIAGNOSED_CASES_FOLDER, selected_file))
            st.info(f"Loaded conversation from: {selected_file}")
        else:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

    # --- Display chat ---
    for msg in st.session_state["messages"]:
        if msg["role"] == "user" and msg.get("images"):
            with st.chat_message("user"):
                st.write(msg["content"])
                for img in msg["images"]:
                    st.image(img, caption="User image", use_container_width=True)
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

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model = genai.GenerativeModel(MODEL_NAME)

        # --- Clean temporary directories ---
        clean_folder(UPLOADED_IMAGES_FOLDER)
        clean_folder(TEMP_QUERY_DATA)

        # --- Save uploaded images ---
        image_paths = []
        if uploaded_files:
            for f in uploaded_files:
                path = save_uploaded_image(f)
                image_paths.append(path)

        # --- Add user message to session ---
        user_msg = {"role": "user", "content": prompt or ""}
        if image_paths:
            user_msg["images"] = image_paths
        st.session_state["messages"].append(user_msg)

        # --- Display in chat ---
        with st.chat_message("user"):
            if prompt:
                st.write(prompt)
            if image_paths:
                for img in image_paths:
                    st.image(img, caption="Uploaded image", use_container_width=True)

        # --- Initialize embedding vectors ---
        text_vector = None
        image_vector = None

        with QueryOrchestrator(session_id=session_id) as orchestrator:
            # --- Text embedding ---
            if prompt:
                st.info("Computing text embedding...")
                try:
                    text_vector = orchestrator.embed_text_query(prompt)
                    st.success(f"Generated text embedding of shape {text_vector.shape}")
                except Exception as e:
                    st.error(f"Text embedding failed: {e}")
                    text_vector = None

            # --- Image embedding ---
            if image_paths:
                for img_path in image_paths:
                    st.info(f"Embedding image: {os.path.basename(img_path)}")
                    try:
                        caption = os.path.splitext(os.path.basename(img_path))[0]
                        image_vector = orchestrator.embed_image_query(img_path, caption_text=caption)
                        st.success(f"Computed image embedding of shape {image_vector.shape}")
                    except Exception as e:
                        st.error(f"Error embedding image {os.path.basename(img_path)}: {e}")
                        image_vector = None

        # --- Step 1: Query Qdrant ---
        st.info("Querying Qdrant for similar cases...")

        try:
            # Call refactored run_query with both modalities
            retrieved_cases, saved_dir = run_query(
                text_vector=text_vector.tolist() if text_vector is not None else None,
                image_vector=image_vector.tolist() if image_vector is not None else None,
                top_k=5,
                session_id=session_id
            )

            st.success(f"Retrieved {len(retrieved_cases)} cases from Qdrant.")
            st.caption(f"Results saved in: {saved_dir}")

        except Exception as e:
            st.error(f"Qdrant query failed: {e}")
            retrieved_cases, saved_dir = {}, None

        # --- Step 2: Prompt Generation + Gemini ---
        with st.spinner("Building prompt and generating analysis..."):
            final_prompt, decoded_images_path = generate_prompt(prompt, session_id=session_id)

            # Build multimodal content
            content_parts = [{"text": final_prompt}]
            for img_path in decoded_images_path:
                try:
                    with open(img_path, "rb") as f:
                        data = f.read()
                    content_parts.append({
                        "inline_data": {"mime_type": "image/png", "data": data}
                    })
                except Exception as e:
                    st.warning(f"Could not attach image {img_path}: {e}")

            # Generate with Gemini
            try:
                response = model.generate_content(content_parts)
                ai_text = response.text.strip()

                # Try parsing Gemini output as JSON (since prompt enforces JSON format)
                try:
                    ai_output = json.loads(ai_text)
                    st.success("Gemini output parsed as valid JSON.")
                except json.JSONDecodeError:
                    st.warning("Gemini output was not valid JSON. Saving raw text instead.")
                    ai_output = {"raw_output": ai_text, "error": "Invalid JSON format"}

            except Exception as e:
                st.error(f"Error calling Gemini: {e}")
                ai_output = {"raw_output": f"Error during model generation: {e}"}
                ai_text = str(ai_output)


        # --- Debug: Show full Gemini prompt content ---
        st.caption(f"Prompt built from retrieved cases in: {saved_dir}")
        with st.expander("🔍 View Full Gemini Prompt (Debug Mode)"):
            st.text_area("Final Gemini Prompt:", final_prompt, height=400)

        # --- Step 3: Display AI response ---
        st.session_state["messages"].append({"role": "assistant", "content": ai_text})
        st.chat_message("assistant").write(ai_text)

        # --- Step 4: Save diagnostic record as JSON ---
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"diagnosis_{timestamp}.json"
        output_path = os.path.join(DIAGNOSED_CASES_FOLDER, filename)

        diagnostic_record = {
            "timestamp": timestamp,
            "user_query": {
                "text": prompt,
                "images": image_paths if image_paths else [],
            },
            "has_image": bool(image_paths),
            "retrieved_cases": retrieved_cases if isinstance(retrieved_cases, dict) else {},
            "generated_prompt": final_prompt,
            "ai_response": ai_output,   # full structured Gemini JSON
            "diagnosis": ai_output.get("analysis", None),
            "correct": None  # can be filled manually for evaluation
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(diagnostic_record, f, indent=4, ensure_ascii=False)
            st.sidebar.success(f"🩺 Diagnosis saved: {filename}")
        except Exception as e:
            st.sidebar.error(f"Error saving diagnostic record: {e}")


# --- Run app ---
if __name__ == "__main__":
    main()
