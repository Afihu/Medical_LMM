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

# Custom modules
from scripts.query import run_query
from scripts.prompt_generate import generate_prompt
from scripts.streamlit_ui_helper import load_conversation, save_conversation

# --- Path configuration ---
# Get the project root (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CASES_FOLDER = os.path.join(PROJECT_ROOT, "cases")
RESPONSES_FOLDER = os.path.join(PROJECT_ROOT, "responses")

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
    os.makedirs(CASES_FOLDER, exist_ok=True)

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
        st.chat_message(msg["role"]).write(msg["content"])

    # --- Chat input ---
    if prompt := st.chat_input("Enter new patient's symptoms and history..."):
        model = genai.GenerativeModel(MODEL_NAME)
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Step 1: Query Qdrant
        query_vector = [0.43, -0.45, 0.78, 0.72]  # placeholder
        run_query(query_vector)

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
