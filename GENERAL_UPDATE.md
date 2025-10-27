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
Each point in that collection can contain both, one, or even neither vector. But upserting a point replaces its vector block entirely. This means:
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