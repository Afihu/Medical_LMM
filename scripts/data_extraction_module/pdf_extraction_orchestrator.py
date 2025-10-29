"""
PDF Extraction Orchestrator (Step 2 & 3)
Coordinates text, image, and caption extraction from PDFs and saves to structured directories.
"""

import os
import json
import re
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Resolve project root (go up from scripts/embedding_mode to root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

from text_extract import extract_text, txt_print
from image_extract import extract_image
from caption_extract import extract_captions
from scripts.config.embedding_config import CONFIG, PATHS, NAMING_CONVENTIONS


class PDFExtractionOrchestrator:
    """Orchestrates extraction of text, images, and captions from PDFs."""
    
    def __init__(self, source_dir=None, output_base_dir=None):
        """
        Initialize the orchestrator.
        
        Args:
            source_dir: Directory containing PDF files (default: from config, resolved from project root)
            output_base_dir: Base directory for outputs (default: project root)
        """
        # Resolve paths from project root
        if output_base_dir is None:
            output_base_dir = PROJECT_ROOT
        
        self.source_dir = source_dir or os.path.join(PROJECT_ROOT, PATHS["source_materials"])
        self.output_base = output_base_dir
        
        # Ensure directories exist
        self._ensure_directories()
        
        self.image_counters = {}  # Track per-case image IDs
        self.caption_counters = {}  # Track per-case caption IDs
    
    @staticmethod
    def _extract_case_id_from_filename(pdf_name):
        """
        Extract case ID from PDF filename.
        
        Expects format: {number}---{description}.pdf or {number}_{description}.pdf
        
        Args:
            pdf_name: Name of the PDF file (without path)
            
        Returns:
            int: Case ID extracted from filename, or None if no valid case number found
        """
        # Match leading digits before separators (---, _, or -)
        match = re.match(r'^(\d+)[_\-]', pdf_name)
        if match:
            return int(match.group(1))
        return None
        
    def _ensure_directories(self):
        """Ensure all output directories exist."""
        for path_key, path_val in PATHS["extracted_data"].items():
            if isinstance(path_val, str):
                full_path = os.path.join(self.output_base, path_val)
                os.makedirs(full_path, exist_ok=True)
    
    def process_pdf(self, pdf_path):
        """
        Process a single PDF file and extract text, images, and captions.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            dict: Extraction results with paths to saved files
        """
        pdf_name = os.path.basename(pdf_path)
        
        # Extract case ID from filename
        case_id = self._extract_case_id_from_filename(pdf_name)
        if case_id is None:
            print(f"\n[ERROR] Could not extract case ID from filename: {pdf_name}")
            print(f"        Expected format: {{number}}---{{description}}.pdf")
            return {
                "case_id": None,
                "pdf_name": pdf_name,
                "text": None,
                "images": [],
                "captions": None,
                "errors": [f"Invalid filename format: {pdf_name}"]
            }
        
        print(f"\n{'='*60}")
        print(f"Processing PDF: {pdf_name} (Case ID: {case_id:03d})")
        print(f"{'='*60}")
        
        results = {
            "case_id": case_id,
            "pdf_name": pdf_name,
            "text": None,
            "images": [],
            "captions": None,
            "errors": []
        }
        
        try:
            # Extract text
            print("\n[1/3] Extracting text sections...")
            text_data = self._extract_and_save_text(pdf_path, case_id)
            if text_data:
                results["text"] = text_data
                print(f"[OK] Text extracted and saved")
            
            # Extract images
            print("\n[2/3] Extracting images...")
            image_paths = self._extract_and_save_images(pdf_path, case_id)
            if image_paths:
                results["images"] = image_paths
                print(f"[OK] Extracted {len(image_paths)} image(s)")
            
            # Extract captions
            print("\n[3/3] Extracting captions...")
            caption_data = self._extract_and_save_captions(pdf_path, case_id)
            if caption_data:
                results["captions"] = caption_data
                captions_count = len(caption_data.get("captions", []))
                print(f"[OK] Extracted {captions_count} caption(s)")
            else:
                print(f"[WARN] No captions found or extraction skipped")
            
        except Exception as e:
            error_msg = f"Error processing {pdf_name}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            results["errors"].append(error_msg)
        
        return results
    
    def _extract_and_save_text(self, pdf_path, case_id):
        """Extract text sections and save to JSON."""
        try:
            text_data = extract_text(pdf_path)
            
            if text_data:
                filename = NAMING_CONVENTIONS["text"].format(case_id=case_id)
                output_path = os.path.join(self.output_base, PATHS["extracted_data"]["text"], filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(text_data, f, indent=2, ensure_ascii=False)
                
                return {"path": output_path, "data": text_data}
        except Exception as e:
            print(f"  [ERROR] Text extraction failed: {e}")
        
        return None
    
    def _extract_and_save_images(self, pdf_path, case_id):
        """Extract images and save as PNG files (staged format)."""
        image_paths = []
        
        try:
            images = extract_image(pdf_path)
            
            if not images:
                print("  No images found in PDF")
                return image_paths
            
            # Initialize counter for this case
            if case_id not in self.image_counters:
                self.image_counters[case_id] = 0
            
            for img_array in images:
                self.image_counters[case_id] += 1
                image_id = self.image_counters[case_id]
                
                # Save as PNG file (staging format)
                filename = NAMING_CONVENTIONS["image"].format(case_id=case_id, image_id=image_id)
                output_path = os.path.join(self.output_base, PATHS["extracted_data"]["images"], filename)
                
                # Convert normalized [-1, 1] array back to [0, 255] uint8 for PNG storage
                img_uint8 = ((img_array + 1.0) * 127.5).astype(np.uint8)
                
                # Save as PNG using PIL
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(img_uint8, 'RGB')
                pil_img.save(output_path)
                
                image_paths.append(output_path)
                print(f"  Saved image {image_id:03d}: {filename}")
        
        except Exception as e:
            print(f"  [ERROR] Image extraction failed: {e}")
        
        return image_paths
    
    def _extract_and_save_captions(self, pdf_path, case_id):
        """Extract captions and save to JSON."""
        try:
            # First extract all text to get captions
            text = ""
            import fitz
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text("text")
            doc.close()
            
            captions = extract_captions(text)
            
            if captions:
                # Group captions into JSON structure
                caption_data = {
                    "total": len(captions),
                    "captions": []
                }
                
                if case_id not in self.caption_counters:
                    self.caption_counters[case_id] = 0
                
                for caption_text in captions:
                    self.caption_counters[case_id] += 1
                    caption_id = self.caption_counters[case_id]
                    
                    caption_data["captions"].append({
                        "id": caption_id,
                        "text": caption_text
                    })
                
                # Save to JSON
                filename = NAMING_CONVENTIONS["caption"].format(case_id=case_id, caption_id=1)
                # Use generic naming for all captions from this case
                filename = f"case_{case_id:03d}_captions.json"
                output_path = os.path.join(self.output_base, PATHS["extracted_data"]["captions"], filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(caption_data, f, indent=2, ensure_ascii=False)
                
                return {"path": output_path, "data": caption_data}
        
        except Exception as e:
            print(f"  [ERROR] Caption extraction failed: {e}")
        
        return None


def process_all_pdfs(source_dir=None, output_base_dir=None):
    """
    Process all PDFs in a directory.
    
    Args:
        source_dir: Directory containing PDFs (default: from config, resolved from project root)
        output_base_dir: Base output directory (default: project root)
        
    Returns:
        list: List of results for each PDF processed
    """
    if output_base_dir is None:
        output_base_dir = PROJECT_ROOT
    
    source_dir = source_dir or os.path.join(PROJECT_ROOT, PATHS["source_materials"])
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist")
        return []
    
    # Find all PDF files
    pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        return []
    
    print(f"\nFound {len(pdf_files)} PDF file(s)")
    
    orchestrator = PDFExtractionOrchestrator(source_dir, output_base_dir)
    results = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(source_dir, pdf_file)
        result = orchestrator.process_pdf(pdf_path)
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs processed: {len(results)}")
    
    # Count as successful if at least text was extracted
    successful = sum(1 for r in results if r["text"] is not None)
    print(f"Successful: {successful}/{len(results)}")
    
    if successful == len(results):
        print("[OK] All PDFs processed successfully")
    else:
        print("\nFailed or partially processed PDFs:")
        for result in results:
            if result["text"] is None:
                status = "[FAIL]"
                print(f"  {status} {result['pdf_name']}: {result['errors'][0] if result['errors'] else 'No text extracted'}")
            elif result["errors"]:
                status = "[WARN]"
                print(f"  {status} {result['pdf_name']}: {result['errors'][0]}")
    
    return results


if __name__ == "__main__":
    # Example usage
    results = process_all_pdfs()
