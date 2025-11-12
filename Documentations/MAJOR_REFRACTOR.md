# Proposed New Structure:
- The entire system will have two pipelines: "Embedding Pipeline" and "Main Pipeline", each with its own orchestration.
- A script called `./scripts/utils/model_manager.py` will be responsible for managing the different models used in both pipelines, using Singleton pattern to ensure only one instance of each model is loaded at any time.

## Embedding Pipeline
- This pipeline will handle the extraction of text and images from PDFs, generation of embeddings, and uploading to Qdrant.
- All of the orchestrators (pdf_extraction_orchestrator, embedding_orchestrator) will be summed up in a main script located at `./scripts/embedding_pipeline_runner.py`.
    - The main script will acts as a CLI interface to run the entire embedding pipeline. Example of the expected interface:
```bash
    uv run "/scripts/embedding_pipeline_runner.py"
    
    #Echo from the script
    What module do you want to run? 
    1. PDF Extraction Module
    2. Embedding Generation Module
    3. Qdrant Upload Service
    4. Exit
    Please enter the number corresponding to your choice:

    # Take user's input and run the selected module or exit

    # After completion, ask if the user wants to run the next module:
    # Embedding Pipeline after completing PDF Extraction Module
    # OR Qdrant Upload Service after completing Embedding Generation Module
    Do you want to run Embedding Generation Module next? (y/n):

    # If yes, proceed to run the next module
    # If no, exit the script. 

```

1. **Data Extraction Module**:
- Location: `./scripts/data_extraction_module/`
- This module has 3 sub-modules: Text Extractor (`text_extract.py`), Image Extractor (`image_extract.py`), and Caption Extractor (`caption_extract.py`).

- Input: PDF files from `./source_materials/`
    
    - Each PDF will be processed to extract:
        - Summarized text sections (Title, History, Clinical Findings, Discussion, Summary Box First Line) by `./scripts/data_extraction_module/text_extract.py` as JSON files, then stored in `extracted_data/text/`.
            - Naming convention: `case_<case_id>_text.json`
            - Expected format:
                ```json
                {
                    "Title": "...",
                    "History": "...",
                    "Clinical Findings": "...",
                    "Discussion": "...",
                    "Summary Box First Line": "..."
                }
                ```
        
        - Images extracted, resized to 448x448 pixels, and normalized to (-1, 1) by `./scripts/data_extraction_module/image_extract.py` as .png files, then stored in `extracted_data/images/`.
            - Naming convention: `case_<case_id>_image_<image_id>.png`
        
        - Captions extracted by `./scripts/data_extraction_module/caption_extract.py` as JSON files, then stored in `extracted_data/captions/`.
            - Naming convention: `case_<case_id>_caption_<caption_id>.json`
            - Expected format:
                ```json
               {
                "total": 2,
                "captions": [
                    {
                    "id": 1,
                    "text": "..."
                    },
                    {
                    "id": 2,
                    "text": "..."
                    }
                    ]
                }
                ```
        - Note that `<case_id>`, `<image_id>`, and `<caption_id>` will start from 001 and increment for each new case, image, or caption.

- Output: Extracted text sections, images, and captions saved in a temporary .json or .png format in `extracted_data/`. Each type of data (text, image, caption) will be stored in separate directories `./text/`, `./images/`, `./captions/`.

2. **Embedding Generation Module**:
- Location: `./scripts/embedding_generation_module/`
- This module has 2 sub-modules: Text Embedding Generator (`text_embedding_generator.py`), Image Embedding Generator (`image_embedding_generator.py`) and a main orchestration file in this pipeline `embedding_orchestrator.py`.
    - **Note**: 
        - Both embedding generators can be reused for the Main Pipeline as well.
        - The embedding dimensions are standardized via `./scripts/config/embedding_config.py`.
        - The `./scripts/utils/model_manager.py` will handle selecting which device (CPU/GPU) to load the models onto and sequentially handling the text and image embedding generation to optimize memory usage.

- Input: Extracted JSON array and .png files from `extracted_data/`
    - Text files will be processed to generate text embeddings, one case per embedding. Stored in `./staged_embeddings/text_embeddings`.

    - One image will be processed to generate image embeddings, one image per embedding. Stored in `./staged_embeddings/image_embeddings`.
        - Tensor-to-NumPy Conversion will be handled here, if needed.

    - One caption per image will also be processed to generate caption embeddings, one caption per embedding. Stored in `./staged_embeddings/caption_embeddings`.

    - Embedding dimensions size and staged embeddings' location are specified in the `./scripts/config/embedding_config.py` file.

    - Naming convention for staged embeddings:
        - Text Embeddings: `case_<case_id>_text_embedding.npy`
        - Image Embeddings: `case_<case_id>_image_<image_id>_embedding.npy`
        - Caption Embeddings: `case_<case_id>_caption_<caption_id>_embedding.npy`

- Output: Staged embeddings are ready to be uploaded to Qdrant.

3. **Qdrant Upload Service**:
- Location: `./scripts/qdrant_services/upload.py`

- Input: Embeddings generated from the Embedding Generation Module. This service will make the connection and upload data to Qdrant.
    - All embeddings will be uploaded to Qdrant with summarized text from the corresponding case (from `extracted_data/text/`) as payload.
    - Qdrant will auto-generate UUIDs for each point (independent in each collection).
    - The `case_id` (extracted from each embedding file's name) in the payload is used to match related embeddings across collections.
    - Point structure in Qdrant:
    ```json
    # For text embeddings (collection: medical_case_texts)
    {
        "id": "<auto-generated-uuid>",
        "vector": [<text_embedding_vector>],
        "payload": {
            "case_id": 001,
            "Title": "...",
            "History": "...",
            "Clinical_Findings": "...",
            "Discussion": "...",
            "Summary_Box_First_Line": "..."
        }
    }

    # For image embeddings (collection: medical_case_images)
    # Note: Multiple points can have the same case_id if a case has multiple images
    {
        "id": "<auto-generated-uuid>",
        "vector": {
            "image": [<image_embedding_vector>],
            "caption": [<caption_embedding_vector>]
        },
        "payload": {
            "case_id": 001,
            "image_id": 001,
            "Caption": "Figure 1: Clinical findings...",
            "Title": "...",
            "History": "...",
            "Clinical_Findings": "...",
            "Discussion": "...",
            "Summary_Box_First_Line": "..."
        }
    },
    {
        "id": "<auto-generated-uuid>",
        "vector": {
            "image": [<image_embedding_vector>],
            "caption": [<caption_embedding_vector>]
        },
        "payload": {
            "case_id": 001,
            "image_id": 002,
            "Caption": "Figure 2: Additional findings...",
            "Title": "...",
            "History": "...",
            "Clinical_Findings": "...",
            "Discussion": "...",
            "Summary_Box_First_Line": "..."
        }
    }
    ```
    - Each type of embedding (text or image) will be stored in its own collection in Qdrant: `medical_case_texts` and `medical_case_images`.
    - UUIDs are independently generated by Qdrant; cross-collection matching is done via `case_id` in payload.

- Output: Embeddings uploaded to Qdrant with appropriate payloads.
    
---

## Main Runtime Pipeline
- This pipeline is the runtime query handler that will handle querying Qdrant based on user input and generating prompts for the LLM. This is intended to be used in the Streamlit app.
- Data generated/staged in this mode will be temporary and stored in `./temp_query_data/` and cleaned up before each session.
- The pipeline consists of 4 main modules:

1. **Main Handler Module**:
    - This module acts as the orchestrator, sitting between the Streamlit app and the other backend modules, handling the communication, data flow and cleanup before each session.
    - **Note**: This module is currently being embedded into the Streamlit app located at `./main_streamlit.py`. In the future, it can be refactored into its own script.

    - Input: User query (text and/or image) from Streamlit app.
        - Text queries will be staged as JSON arrays.
        - Images uploaded will be resized and normalized in a similar fashion to `./scripts/data_extraction_module/image_extract.py` as .png files inside `./temp_query_data/`.
        - Expected naming convention:
            - Text Query: `user_query_text_<timestamp>.json`
            - Image Query: `user_query_image_<timestamp>.png`
    
    - Output: Final answer from LLM.
        
        - For evaluation purposes, the final answer and retrieved cases can be logged in `./diagnosed_cases/` with timestamped filenames.
            - Example structure:
                - File name: `diagnosed_case_2024-10-01_15-30-00.json`
                ```json
                {
                    "timestamp": "2024-10-01_15-30-00",
                    # User Input
                    "user_query": "...",
                    "has_image": true,
                    # Context
                    "retrieved_cases": {
                        "case_<case_id>": # This is extracted from the retrieved cases' payload
                        {
                            "title": "...",
                            "history": "...",
                            "clinical_findings": "...",
                            "discussion": "...",
                            "summary_box_first_line": "...",
                        },
                        ...
                    },
                    # Response
                    "ai-response": {
                        "analysis": "...",
                        "differential_diagnosis": [...]
                    }
                }
                ```

2. **Embedding Query Module**:
   - This module is the same as Embedding Generation Module.
   - Input: The staged user query (text or/and image).
        - Each data type will be processed to generate temporary embeddings with their respective embedding generators. 
        - The location of staged embeddings is `./temp_query_data/embeddings/`.
        - Naming convention:
            - Text Query Embedding: `user_query_text_embedding_timestamp.npy`
            - Image Query Embedding: `user_query_image_embedding_timestamp.npy`
   - Output: Temporary embeddings saved for querying against Qdrant.

3. **Qdrant Query Module**:
    - Top-k values for retrieval will be specified in `./scripts/config/query_config.py`. 
        - `text_top_k` and `image_top_k` can be set independently.

    - Input: Temporary embeddings from the Embedding Query Module.
        - Query Logic:
            - Perform similarity search in Qdrant collections (`medical_case_texts` and/or `medical_case_images`) based on what embeddings are available from embedding module.
            - After retrieving top-k similar cases from each collection:
                - If both modalities are present, merge results based on `case_id` in payload to avoid duplicates.
                - If only one modality is present, use results from that modality directly.

    - Output: Retrieved top-k relevant cases from Qdrant.

4. **Prompt Generation Module**:
    - This module will format the retrieved cases and user query into a structured prompt for the LLM.
    - After this, the answer handed back to the Main Handler Module for final output to the user.
    - Input: Retrieved top-k cases from Qdrant, user original query and system prompt for LLM.
    - Output: Final prompt for LLM and sent to the Gemini API.

## LLM Wrapper Module
- Rationale: As we will be using both local hosted models and external APIs for LLM services, a unified wrapper will be created to handle interactions with different LLM providers.
- Location: `./scripts/llm_services/*`
- This module will contain different classes for each LLM provider (e.g. gemini_provider.py, lmstudio_provider.py) that implement a common interface for sending prompts and receiving responses.
- To further facilitate this change, a factory pattern will be implemented in `llm_factory.py` to instantiate the appropriate LLM provider based on configuration settings.
    - Gemini API will follow the current implementation (as in `main_streamlit.py`)
    - For local models, `LM Studio` uses OpenAI-like API structure, so the wrapper will adapt accordingly.

- Input: Final prompt from the Prompt Generation Module.
- Output: LLM response sent back to the Main Handler Module.

**Plan for LLM Wrapper Module Transition:**
- Decouple the current Gemini API calls in `main_streamlit.py` into a separate class in `./scripts/llm_services/gemini_provider.py`.
- Implement the factory pattern in `llm_factory.py` to allow easy switching between Gemini and LM Studio.
- Implement `lmstudio_provider.py` to handle local model interactions (as specified in `LM_STUDIO_IMPLEMENTATION_SUMMARY.md` and `LLM_WRAPPER_ANALYSIS.md`)