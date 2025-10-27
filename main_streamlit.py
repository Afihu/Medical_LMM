"""
main_streamlit.py
-----------------
Streamlit-based version of the Medical_LMM chatbot.
This app connects to Gemini and Qdrant to provide contextual medical analysis.

Run with:
    streamlit run main_streamlit.py
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# Import local modules
from scripts.query import run_query
from scripts.prompt_generate import generate_prompt

# --- Streamlit Page Setup ---
st.set_page_config(page_title="🩺 Medical LMM Chat", page_icon="💬", layout="centered")
st.title("🩺 Medical Diagnostic Assistant")
st.caption("🚀 Powered by Gemini + Qdrant + CLIP")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Enter your Gemini API key:")
    gemini_api_key = st.text_input("Gemini API Key", key="gemini_api_key", type="password")
    st.markdown("[Get an API key](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.markdown("**Project Info:**")
    st.markdown("• Data retrieved from Qdrant Cloud\n• Prompt composed from retrieved cases")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello 👋! Describe your symptoms and I’ll analyze them based on similar medical cases."}
    ]

# --- Display Chat History ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Setup Gemini ---
def setup_gemini(api_key: str):
    if not api_key:
        st.warning("Please provide your Gemini API key in the sidebar to continue.")
        st.stop()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    return model

# --- Chat Input ---
if user_input := st.chat_input("Enter patient symptoms or medical history..."):
    if not gemini_api_key:
        st.info("Please add your Gemini API key to continue.")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Step 1: Generate CLIP embedding (placeholder)
    query_vector = [0.12, -0.45, 0.78, 0.66]

    # Step 2: Query Qdrant
    with st.spinner("🔍 Searching for similar cases..."):
        run_query(query_vector)

    # Step 3: Generate prompt from retrieved cases
    with st.spinner("🧠 Building contextual prompt..."):
        final_prompt = generate_prompt(user_input)

    # Step 4: Send to Gemini API
    with st.spinner("🤖 Generating diagnostic report..."):
        try:
            model = setup_gemini(gemini_api_key)
            response = model.generate_content(final_prompt)
            ai_message = response.text

            # Append to chat and display
            st.session_state.messages.append({"role": "assistant", "content": ai_message})
            st.chat_message("assistant").write(ai_message)

            # Step 5: Save to Markdown log
            responses_folder = os.path.join(os.path.dirname(__file__), "responses")
            os.makedirs(responses_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.md"
            filepath = os.path.join(responses_folder, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"User: {user_input}\n\nAI: {ai_message}\n")

            st.toast(f"💾 Saved conversation to {filename}")

        except Exception as e:
            st.error(f"Error during Gemini generation: {e}")
