# Update 19.11.2025
Added evaluation modes for evaluation pipeline. There are 3 different modes:
- "internal": LLM uses only internal knowledge, no RAG retrieval
- "rag": LLM uses only retrieved context, no internal knowledge (RAG-only)
- "hybrid": LLM uses both internal knowledge + retrieved context (default)

For running the evaluation:
- **text only**: run quick_eval.py, read the readme in the batch eval branch for more options
- **text and image**: run UI like usual, input text and image there and then once you confirm diagnosed cases are present in the file directory, then run quick_eval but add in the argument to skip diagnosing (once again, refer to the [Batch Evaluation](/evaluate/BATCH_EVAL_README.md)) 

# Update 4.11.2025
The refactoring for the backend is done. App can now work normally (with Streamlit too!).

# Update 27.10.2025 (Part 2)
You can now interact with the system via main_streamlit.py, which makes use of Streamlit, an open-source library for data science project.

This only supports chatting with the model, not yet for uploading cases (will be added later).

To run the project, run in the terminal:
```bash
uv run streamlit run main_streamlit.py
```

# Update 27.10.2025
Sorry for the long wait, but here is the upload_cases_image.py and upload_cases_text.py

With both of the files, the backend is now functionally complete.

Feel free to remind me for adjustment!

# Update 23.10.2025
Turns out Qdrant does not support partial vector updates.

When you create a collection with multiple vector fields, e.g.:
```python
vectors_config={
    "text_vec": VectorParams(size=512, distance=Distance.COSINE),
    "image_vec": VectorParams(size=512, distance=Distance.COSINE)
}
```
Each point in that collection can contain both, one, or even neither vector. But upserting a point replaces its vector block entirely. 

This means:
- Upsert either { "text_vec": [...] } or { "image_vec": [...] } for the first time is fine
- If later someone upserts { "image_vec": [...] } without including text_vec, Qdrant will remove the old text_vec, because the new upsert overwrote the entire vector dictionary for that point.

## Solution
Two separate scripts for uploading text vectors and image vectors will be made. 

This will create 2 different Collections for cases (either text_vec or image_vec, or both)

## Motivation for the solution
- Thanh has just found out that there are some cases that only contains texts, without images, and vice versa. Therefore, it is optimal to split into 2 Collections for querying.
- We will query the Qdrant DB in accordance to the data type of the user's prompt (either text or image, or both). This also support the conclusion for the first point.

# Update 22.10.2025
I cannot think of a way except pushing all the content from the previous run, which is inefficient.

I will try to figure it out later 

# Update 21.10.2025
- Scripts for uploading embedded vectors are done. Usable now
- Conversation with the model is limited with one-time prompting. This is due to the fact that we are using the terminal and we do not know if Gemini can retain the memory of the chat after we close the terminal.
**Note**: Turns out it does. We will upgrade the chatting methods into few-shot prompting. ETA: 4pm 22.10.2025