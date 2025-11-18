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
from dotenv import load_dotenv
import numpy as np
import json
import re
import base64

# LLM Services (New wrapper)
from scripts.llm_services import get_llm_provider
from scripts.llm_services.content_adapter import ContentAdapter, to_provider_format, add_medical_context
from scripts.llm_services.factory import setup_provider_from_env, validate_provider_config

# Custom modules
from scripts.qdrant_services.query import run_query, normalize_cases_for_output
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

# --- Global LLM Provider ---
llm_provider = None

# --- Setup ---
def setup():
    """Configure LLM provider and ensure directories exist."""
    global llm_provider
    
    load_dotenv()
    
    # Check which provider is configured
    provider_type = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    # Validate provider configuration
    is_valid, error_msg = validate_provider_config(provider_type)
    if not is_valid:
        st.error(f"LLM Provider configuration error: {error_msg}")
        st.info("Please check your .env file and ensure the required environment variables are set.")
        st.stop()
    
    # Setup LLM provider
    try:
        llm_provider = setup_provider_from_env()
        st.success(f"✅ {llm_provider.get_provider_name().title()} provider initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize LLM provider: {e}")
        st.stop()
    
    # Create necessary directories
    os.makedirs(DIAGNOSED_CASES_FOLDER, exist_ok=True)
    os.makedirs(UPLOADED_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(TEMP_QUERY_DATA, exist_ok=True)
    os.makedirs(os.path.join(TEMP_QUERY_DATA, "user_query"), exist_ok=True)

def parse_json_output(text):
    """
    Legacy function - now handled by LLM providers.
    This function is kept for backwards compatibility but delegates to the active provider.
    """
    global llm_provider
    if llm_provider:
        return llm_provider.parse_response(text)
    else:
        # Fallback to basic JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "raw_output": text,
                "error": "Could not parse JSON - no LLM provider available",
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }

def extract_json(text):
    """
    Legacy function - now handled by LLM providers.
    This function is kept for backwards compatibility.
    """
    global llm_provider
    if llm_provider and hasattr(llm_provider, 'extract_json'):
        return llm_provider.extract_json(text)
    else:
        # Fallback to basic extraction
        match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
        if match:
            return match.group(1).strip()
        return text.strip()

# --- Main Loop ---
def main():
    setup()
    st.set_page_config(page_title="Medical LMM", page_icon="💬", layout="wide")
    st.title("💬 Medical LMM Diagnostic Assistant")
    st.caption("AI-assisted medical case reasoning using Gemini + Qdrant")

    # --- Sidebar: history ---
    # with st.sidebar:
    #     st.header("🗂️ Chat History")
    #     response_files = sorted(
    #         [f for f in os.listdir(DIAGNOSED_CASES_FOLDER) if f.endswith(".md")],
    #         reverse=True
    #     )
    #     selected_file = st.selectbox("Select a past conversation:", ["(New Chat)"] + response_files)

    #     if selected_file != "(New Chat)":
    #         st.session_state["messages"] = load_conversation(os.path.join(DIAGNOSED_CASES_FOLDER, selected_file))
    #         st.info(f"Loaded conversation from: {selected_file}")
    #     else:
    #         st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
    with st.sidebar:
        st.header("🗂️ Chat History")

        response_files = sorted(
            [f for f in os.listdir(DIAGNOSED_CASES_FOLDER) if f.endswith(".json")],
            reverse=True
        )

        selected_file = st.selectbox(
            "Select a past conversation:",
            ["New Chat"] + response_files,
            key="selected_case"
        )


        if selected_file != "New Chat":
            file_path = os.path.join(DIAGNOSED_CASES_FOLDER, selected_file)

            # Load saved diagnosed case JSON
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Build a chat-style history ----------------------------------------
            messages = []

            # USER MESSAGE
            user_block = {"role": "user"}

            # user text
            user_text = data.get("user_query", "")
            if isinstance(user_text, dict):
                # if your structure is { "text": "...", "images": [...] }
                user_block["content"] = user_text.get("text", "")
                user_images = user_text.get("images", [])
            else:
                # fallback
                user_block["content"] = str(user_text)
                user_images = []

            # user image (if stored)
            if user_images:
                # user_images is expected to contain base64 strings
                user_block["image"] = user_images[0]  # show only 1 for now

            messages.append(user_block)

            # AI MESSAGE
            ai_resp = data.get("ai_response", {})
            ai_text = ""

            analysis = ai_resp.get("analysis", "")
            ddx = ai_resp.get("differential_diagnosis", [])

            ai_text += "**Analysis:**\n" + analysis + "\n\n"
            ai_text += "**Differential Diagnosis:**\n"
            if isinstance(ddx, list):
                for d in ddx:
                    ai_text += f"- {d}\n"
            else:
                ai_text += str(ddx)

            messages.append({"role": "assistant", "content": ai_text})

            # Save rebuilt chat into session state
            st.session_state["messages"] = messages
            st.info(f"Loaded diagnosed case: {selected_file}")

        # else:
        #     # New chat
            

    # --- Display chat ---
    # for msg in st.session_state["messages"]:
    #     if msg["role"] == "user" and msg.get("images"):
    #         with st.chat_message("user"):
    #             st.write(msg["content"])
    #             for img in msg["images"]:
    #                 st.image(img, caption="User image", use_container_width=True)
    #     else:
    #         st.chat_message(msg["role"]).write(msg["content"])

    

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):

            # text
            st.write(msg.get("content", ""))

            # image (base64)
            if "image" in msg:
                # [DEBUG]: Currently not stor
                # import base64
                # from io import BytesIO
                # from PIL import Image

                # image_b64 = msg["image"]
                # image_bytes = base64.b64decode(image_b64)
                # img = Image.open(BytesIO(image_bytes))

                # st.image(img, caption="User Image", use_column_width=True)
                continue

    if st.session_state.get("selected_case") == "New Chat":   # Only show chat functionalities when in a new chat

        # --- Chat input ---
        # st.divider()
        # col1, col2 = st.columns([7, 2])
        # with col1:
        #     prompt = st.text_input(
        #         "Enter patient's symptoms and history...",
        #         key="chat_input",
        #         label_visibility="collapsed",
        #     )
        # with col2:
        #     uploaded_files = st.file_uploader(
        #         "➕",
        #         type=["jpg", "jpeg", "png"],
        #         label_visibility="collapsed",
        #         key="image_input",
        #         accept_multiple_files=True,  # allow multiple images
        #     )
        st.session_state["messages"] = [{
                "role": "assistant",
                "content": "Hello! How can I assist you today?"
        }]

        uploaded_files = st.file_uploader(
            "➕",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="image_input",
            accept_multiple_files=True,  # allow multiple images
        )

        prompt = st.text_input(
                "Enter patient's symptoms and history...",
                key="chat_input",
                label_visibility="collapsed",
            )
        
        

        # --- Submit button ---
        if st.button("Send", use_container_width=True):
            if not prompt and not uploaded_files:
                st.warning("Please enter text or upload an image.")
                st.stop()

            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure LLM provider is initialized
            if not llm_provider:
                st.error("LLM provider not initialized. Please refresh the page.")
                st.stop()

            # --- Clean temporary directories before each session ---
            # This ensures no residual data from previous sessions (as per plan)
            clean_folder(TEMP_QUERY_DATA)
            os.makedirs(os.path.join(TEMP_QUERY_DATA, "user_query"), exist_ok=True)
            os.makedirs(os.path.join(TEMP_QUERY_DATA, "embeddings"), exist_ok=True)
            os.makedirs(os.path.join(TEMP_QUERY_DATA, session_id), exist_ok=True)

            # --- Save uploaded images ---
            image_paths = []
            query_image_paths = []
            if uploaded_files:
                # Clean all relevant temp folders before new query
                print("\nClearing temporary directories for a new query...\n")
                clean_folder(os.path.join(PROJECT_ROOT, "uploaded_images"))
                clean_folder(os.path.join(PROJECT_ROOT, "decoded_images"))
                clean_folder(os.path.join(PROJECT_ROOT, "cases_image"))
                clean_folder(os.path.join(PROJECT_ROOT, "cases_text"))
                print("\nClearing done! Begin saving uploaded images...\n")
                for f in uploaded_files:
                    display_path, query_path = save_uploaded_image(f, session_id=session_id)
                    image_paths.append(display_path)
                    query_image_paths.append(query_path)

            # --- Add user message to session ---
            user_msg = {"role": "user", "content": prompt or ""}
            if image_paths:
                user_msg["images"] = image_paths
            st.session_state["messages"].append(user_msg)

            # --- Save user query to temp_query_data/user_query/ (per plan) ---
            if prompt:
                # import json
                user_query_dir = os.path.join(TEMP_QUERY_DATA, "user_query")
                os.makedirs(user_query_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                query_filename = f"user_query_text_{timestamp}.json"
                query_path = os.path.join(user_query_dir, query_filename)
                with open(query_path, "w", encoding="utf-8") as f:
                    json.dump({"text": prompt}, f, ensure_ascii=False)

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
                # Call refactored run_query with both modalities (top_k will be read from config)
                retrieved_cases, saved_dir = run_query(
                    text_vector=text_vector.tolist() if text_vector is not None else None,
                    image_vector=image_vector.tolist() if image_vector is not None else None,
                    session_id=session_id
                )

                st.success(f"Retrieved {len(retrieved_cases)} cases from Qdrant.")
                st.caption(f"Results saved in: {saved_dir}")

            except Exception as e:
                st.error(f"Qdrant query failed: {e}")
                retrieved_cases, saved_dir = {}, None

            # --- Step 2: Prompt Generation + LLM Generation ---
            with st.spinner("Building prompt and generating analysis..."):
                final_prompt, decoded_images_path = generate_prompt(prompt, session_id=session_id)

                # Build content for LLM provider
                # Combine reference images and user images
                all_image_paths = decoded_images_path + image_paths
                
                # Add medical context to the content
                content_dict = {
                    "text": final_prompt,
                    "images": all_image_paths if all_image_paths else None
                }
                
                # Add medical system prompt
                content_with_medical_context = add_medical_context(content_dict)
                
                # Convert to provider-specific format
                provider_name = llm_provider.get_provider_name()
                content_for_provider = to_provider_format(
                    content_with_medical_context, 
                    provider_name
                )
                
                # Log attached images
                if decoded_images_path:
                    st.info(f"Attached {len(decoded_images_path)} reference images from retrieved cases")
                if image_paths:
                    for img_path in image_paths:
                        st.info(f"Attached user image: {os.path.basename(img_path)}")

                # Generate with LLM provider
                try:
                    st.info(f"Generating response using {provider_name.title()} provider...")
                    ai_text = llm_provider.generate_content(content_for_provider)
                    
                    # Parse response using provider's parser
                    ai_output = llm_provider.parse_response(ai_text)
                    
                    if "error" not in ai_output:
                        st.success(f"{provider_name.title()} output parsed as valid JSON.")
                    else:
                        st.warning(f"{provider_name.title()} output was not valid JSON. Saving raw text instead.")

                except Exception as e:
                    st.error(f"Error calling {provider_name.title()} provider: {e}")
                    ai_output = {
                        "raw_output": f"Error during model generation: {e}",
                        "error": str(e),
                        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        "analysis": "Failed to generate response",
                        "differential_diagnosis": {},
                        "confidence_level": "unknown"
                    }
                    ai_text = str(ai_output)


            # --- Debug: Show full LLM prompt content ---
            st.caption(f"Prompt built from retrieved cases in: {saved_dir}")
            with st.expander(f"View Full {llm_provider.get_provider_name().title()} Prompt (Debug Mode)"):
                st.text_area(f"Final {llm_provider.get_provider_name().title()} Prompt:", final_prompt, height=400)

            # --- Step 3: Display AI response ---
            st.session_state["messages"].append({"role": "assistant", "content": ai_text})
            st.chat_message("assistant").write(ai_text)

            # --- Step 4: Save diagnostic record as JSON ---
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"diagnosed_case_{timestamp}.json"
            output_path = os.path.join(DIAGNOSED_CASES_FOLDER, filename)

            # Normalize retrieved cases to remove redundancy
            # Keep only: title, history, clinical_findings, discussion, summary_box_first_line
            normalized_cases = normalize_cases_for_output(retrieved_cases) if retrieved_cases else {}

            diagnostic_record = {
                "timestamp": timestamp,
                "user_query": {
                    "text": prompt,
                    "images": query_image_paths if query_image_paths else [],
                },
                "has_image": bool(query_image_paths),
                "retrieved_cases": normalized_cases,
                "ai_response": ai_output,
            }

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(diagnostic_record, f, indent=4, ensure_ascii=False)
                st.sidebar.success(f"Diagnosis saved: {filename}")
            except Exception as e:
                st.sidebar.error(f"Error saving diagnostic record: {e}")

    else:   # for the "New Chat" condition
        st.info("Viewing a previously saved diagnosed case. Input is disabled.")

# --- Run app ---
if __name__ == "__main__":
    main()
