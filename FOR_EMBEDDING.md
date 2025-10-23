# Update 23.10.2025
Turns out Qdrant does not support partial vector updates.
Two separate scripts for uploading text vectors and image vectors will be made. This will create 2 different Collections for a cases (either text_vec or image_vec, or both)
ETA: 2pm 24.10.2025
For more information, see at [General Update](GENERAL_UPDATE.md)

# Update 20.10.2025
We can upload your vectors embeddings now.

Here is the current work flow:
## Step 1
Make sure you have the .env file in your root folder. It should look like this:
```bash
GEMINI_API_KEY=""
QDRANT_URL=""
QDRANT_API_KEY=""
```

## Step 2
Fill the folder /temp_data with the data you like. Make sure the cases are in this format:
```json
{
    "id": "case_001",
    "vector": {
        "text_vector": [...],
        "image_vector": [...]
    },
    "payload": {
        "diagnosis": "...",
        "symptoms": "...",
        "history": "..."
    }
}
```
The file can consist a LIST of cases too.

## Step 3
Run this command from project root:
```bash
uv run scripts/upload_cases.py
```

## Step 4
Check for the result in the terminal. You should see something like this:
```nginx
Connected to Qdrant Cloud: https://...
Uploaded 5 cases to 'star_charts' collection.
```
