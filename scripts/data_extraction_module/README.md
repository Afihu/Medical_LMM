# Embedding Mode - PDF Extraction Pipeline

## Overview

This module implements Steps 1-3 of the stepwise implementation plan for the Medical LMM system.

### Directory Structure

```
scripts/
├── config/
│   ├── __init__.py
│   └── embedding_config.py          # Step 1: Configuration
├── embedding_mode/
│   ├── __init__.py
│   └── pdf_extraction_orchestrator.py  # Step 2 & 3: Orchestration
extracted_data/
├── text/                            # .json files (extracted text sections)
├── images/                          # .npy files (normalized image arrays)
└── captions/                        # .json files (extracted captions)
staged_embeddings/                   # Will store .npy embedding files (Step 4)
```

---

## Step 1: Embedding Dimension Configuration

**File:** `scripts/config/embedding_config.py`

Standardizes embedding dimensions and file naming conventions across all modules.

### Key Features:
- `CONFIG`: Embedding dimensions (1024 for both text and image)
- `PATHS`: Standardized directory structure
- `NAMING_CONVENTIONS`: File naming patterns with ID formatting

### Example Usage:
```python
from scripts.config.embedding_config import CONFIG, PATHS, NAMING_CONVENTIONS

# Access embedding dimension
dim = CONFIG["text_embedding_dim"]  # 1024

# Access output paths
text_dir = PATHS["extracted_data"]["text"]  # "extracted_data/text"

# Generate filename
filename = NAMING_CONVENTIONS["text"].format(case_id=1)
# Output: "case_001_text.json"
```

---

## Step 2 & 3: PDF Extraction Orchestrator

**File:** `scripts/embedding_mode/pdf_extraction_orchestrator.py`

Coordinates extraction of text, images, and captions from PDFs and saves to structured directories.

### Main Class: `PDFExtractionOrchestrator`

#### Initialization:
```python
from scripts.embedding_mode.pdf_extraction_orchestrator import PDFExtractionOrchestrator

orchestrator = PDFExtractionOrchestrator(
    source_dir="source_materials",  # Directory with PDFs
    output_base_dir="./"            # Output base directory
)
```

#### Processing Single PDF:
```python
result = orchestrator.process_pdf("path/to/file.pdf")
# Returns: {
#     "case_id": 1,
#     "pdf_name": "file.pdf",
#     "text": {"path": "...", "data": {...}},
#     "images": ["path1.npy", "path2.npy"],
#     "captions": {"path": "...", "data": {...}},
#     "errors": []
# }
```

#### Processing All PDFs:
```python
from scripts.embedding_mode.pdf_extraction_orchestrator import process_all_pdfs

results = process_all_pdfs(
    source_dir="source_materials",
    output_base_dir="./"
)
```

### Extraction Details

#### Text Extraction
- **Input:** PDF file
- **Output:** `extracted_data/text/case_001_text.json`
- **Format:** JSON with sections:
  ```json
  {
    "Title": "...",
    "History": "...",
    "Clinical Findings": "...",
    "Discussion": "...",
    "Summary Box First Line": "..."
  }
  ```
- **Source Function:** `preprocessor.text_extract.extract_text()`

#### Image Extraction
- **Input:** PDF file
- **Output:** `extracted_data/images/case_001_image_001.png` (staged PNG format)
- **Preprocessing during extraction:**
  - Convert to RGB
  - Resize to 448×448 (MedSigLip-448 requirement)
  - Normalize to [-1.0, 1.0] range (temporarily)
  - Convert back to [0, 255] uint8 for PNG storage
- **Why PNG staging?**
  - Staged files are human-viewable for inspection/debugging
  - Can be reused for other purposes before embedding generation
  - Embedding generation (Step 4) will read `.png` files and convert to `.npy` embeddings
- **Source Function:** `preprocessor.image_extract.extract_image()` (updated)

#### Caption Extraction
- **Input:** PDF file
- **Output:** `extracted_data/captions/case_001_captions.json`
- **Format:** JSON with caption list:
  ```json
  {
    "total": 5,
    "captions": [
      {"id": 1, "text": "Figure 1: ..."},
      ...
    ]
  }
  ```
- **Source Function:** `preprocessor.caption_extract.extract_captions()`

---

## Updated Functions

### `preprocessor/image_extract.py`

Updated `extract_image()` function with:
- ✅ RGB conversion (unchanged)
- ✅ 448×448 resizing using TensorFlow
- ✅ Normalization to [-1.0, 1.0] range
- ✅ Returns float32 arrays ready for MedSigLip-448

**Key Changes:**
```python
# Resize to 448x448 using TensorFlow
resized = tf_resize(image_array, [448, 448], method='bilinear', antialias=False).numpy().astype(np.uint8)

# Normalize to (-1, 1) range
normalized = (resized.astype(np.float32) / 127.5) - 1.0
```

---

## File Naming Conventions

All extracted files follow strict naming patterns for automatic processing in later stages.

### Text Files
```
case_001_text.json          # case_<case_id:03d>_text.json
case_002_text.json
...
```

### Image Files
```
case_001_image_001.npy      # case_<case_id:03d>_image_<image_id:03d>.npy
case_001_image_002.npy
case_002_image_001.npy
...
```

### Caption Files
```
case_001_captions.json      # case_<case_id:03d>_captions.json
case_002_captions.json
...
```

## Pipeline Architecture

### Steps 1-3: Extraction (Staged Format)

```
PDF Files (source_materials/)
         ↓
    Step 1: Config (embedding_config.py)
         ↓
    Step 2&3: Extraction Orchestrator (pdf_extraction_orchestrator.py)
         ├─→ Text Extractor → case_001_text.json
         ├─→ Image Extractor → case_001_image_001.png
         └─→ Caption Extractor → case_001_captions.json
         ↓
    Staged Data (extracted_data/)
```

**Key Point:** Extraction produces **staged files** in standard formats:
- Text: JSON (human-readable structured data)
- Images: PNG (viewable, can inspect before embedding)
- Captions: JSON (human-readable)

### Step 4: Embedding Generation (To Be Implemented)

```
Staged Data (extracted_data/)
         ↓
    Embedding Generator (embedding_generator_orchestration.py)
    - Reads .json text files → generates text embeddings
    - Reads .png image files → generates image embeddings
         ↓
    Embeddings (staged_embeddings/)
    - case_001_text_embedding.npy
    - case_001_image_001_embedding.npy
```

---

### Command Line
```bash
cd scripts/embedding_mode
python pdf_extraction_orchestrator.py
```

### As Module
```python
from scripts.embedding_mode.pdf_extraction_orchestrator import process_all_pdfs

# Process all PDFs in source_materials/
results = process_all_pdfs()

# Check results
for result in results:
    print(f"Case {result['case_id']}: {result['pdf_name']}")
    if result['text']:
        print(f"  ✓ Text saved to {result['text']['path']}")
    if result['images']:
        print(f"  ✓ {len(result['images'])} images saved")
    if result['captions']:
        print(f"  ✓ Captions saved to {result['captions']['path']}")
```

---

## Output Directory Structure After Execution

```
extracted_data/                          # Staged files (Step 2-3)
├── text/
│   ├── case_001_text.json               # JSON with sections
│   ├── case_002_text.json
│   └── ...
├── images/
│   ├── case_001_image_001.png           # PNG (viewable)
│   ├── case_001_image_002.png
│   ├── case_002_image_001.png
│   └── ...
└── captions/
    ├── case_001_captions.json           # JSON with caption list
    ├── case_002_captions.json
    └── ...

staged_embeddings/                       # Embeddings (Step 4, not yet implemented)
├── case_001_text_embedding.npy          # Will be generated from text.json
├── case_001_image_001_embedding.npy     # Will be generated from image_001.png
├── case_002_text_embedding.npy
└── ...
```

---

## Next Steps

**Step 4:** Create `embedding_generator_orchestration.py` to:
- Load extracted data from `extracted_data/`
- Generate embeddings using MedSigLip-448
- Save embeddings to `staged_embeddings/`

---

## Troubleshooting

### "No PDF files found"
- Ensure `source_materials/` directory exists
- Verify PDF files are in the correct location

### Image extraction errors
- Check that PyMuPDF (fitz) is installed
- Verify TensorFlow is installed for resizing

### Missing outputs
- Check permissions on `extracted_data/` directory
- Ensure sufficient disk space
