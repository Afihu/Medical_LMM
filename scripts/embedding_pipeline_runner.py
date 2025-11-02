"""
Embedding Pipeline Runner
Unified CLI interface for the entire embedding pipeline.
Orchestrates: PDF Extraction → Embedding Generation → Qdrant Upload
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.data_extraction_module.pdf_extraction_orchestrator import PDFExtractionOrchestrator, process_all_pdfs
from scripts.embedding_generation_module.orchestrators.embedding_orchestrator import EmbeddingOrchestrator
from scripts.qdrant_services.upload import QdrantUploadService
from scripts.config.embedding_config import CONFIG, PATHS


def print_menu():
    """Print the main menu."""
    print(f"\n{'='*70}")
    print("MEDICAL LMM - EMBEDDING PIPELINE RUNNER")
    print(f"{'='*70}")
    print("\nWhat module do you want to run?")
    print("  1. PDF Extraction Module")
    print("  2. Embedding Generation Module")
    print("  3. Qdrant Upload Service")
    print("  4. Run Full Pipeline (1 → 2 → 3)")
    print("  5. Exit")
    print(f"{'='*70}\n")


def run_pdf_extraction():
    """Run the PDF extraction module."""
    print(f"\n{'='*70}")
    print("STEP 1: PDF EXTRACTION MODULE")
    print(f"{'='*70}\n")
    
    try:
        results = process_all_pdfs()
        
        # Print results
        if results:
            successful = sum(1 for r in results if r["text"] is not None)
            print(f"\n[SUMMARY] {successful}/{len(results)} PDFs processed successfully")
            return True
        else:
            print("[ERROR] No PDFs were processed")
            return False
    
    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        return False


def run_embedding_generation():
    """Run the embedding generation module."""
    print(f"\n{'='*70}")
    print("STEP 2: EMBEDDING GENERATION MODULE")
    print(f"{'='*70}\n")
    
    try:
        orchestrator = EmbeddingOrchestrator(staging_mode="production")
        results = orchestrator.generate_all_embeddings()
        
        # Check results
        total_successful = results["summary"]["text_successful"] + results["summary"]["image_successful"]
        total_files = len(results["text"]) + len(results["images"])
        
        if total_successful == total_files and total_files > 0:
            print(f"\n[SUMMARY] All embeddings generated successfully ({total_files} total)")
            return True
        elif total_successful > 0:
            print(f"\n[SUMMARY] Partial success: {total_successful}/{total_files} embeddings generated")
            return True
        else:
            print("[ERROR] No embeddings were generated")
            return False
    
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_qdrant_upload():
    """Run the Qdrant upload service."""
    print(f"\n{'='*70}")
    print("STEP 3: QDRANT UPLOAD SERVICE")
    print(f"{'='*70}\n")
    
    try:
        service = QdrantUploadService()
        results = service.upload_all_embeddings()
        
        # Check results
        if results["success"]:
            print(f"\n[SUMMARY] All embeddings uploaded successfully ({results['total_uploaded']} total)")
            return True
        elif results["total_uploaded"] > 0:
            print(f"\n[SUMMARY] Partial upload: {results['total_uploaded']}/{results['total_files']} embeddings uploaded")
            return True
        else:
            print("[ERROR] No embeddings were uploaded")
            return False
    
    except Exception as e:
        print(f"[ERROR] Qdrant upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pipeline():
    """Run the complete pipeline: extraction → embedding → upload."""
    print(f"\n{'='*70}")
    print("RUNNING FULL EMBEDDING PIPELINE (1 → 2 → 3)")
    print(f"{'='*70}\n")
    
    # Step 1: PDF Extraction
    print("\n[1/3] Starting PDF Extraction...")
    if not run_pdf_extraction():
        print("\n[ERROR] PDF extraction failed. Pipeline stopped.")
        return False
    
    # Ask to continue
    response = input("\n[PROMPT] PDF extraction complete. Run Embedding Generation? (y/n): ").strip().lower()
    if response != 'y':
        print("[INFO] Pipeline stopped by user.")
        return False
    
    # Step 2: Embedding Generation
    print("\n[2/3] Starting Embedding Generation...")
    if not run_embedding_generation():
        print("\n[ERROR] Embedding generation failed. Pipeline stopped.")
        return False
    
    # Ask to continue
    response = input("\n[PROMPT] Embedding generation complete. Run Qdrant Upload? (y/n): ").strip().lower()
    if response != 'y':
        print("[INFO] Pipeline stopped by user.")
        return False
    
    # Step 3: Qdrant Upload
    print("\n[3/3] Starting Qdrant Upload...")
    if not run_qdrant_upload():
        print("\n[ERROR] Qdrant upload failed.")
        return False
    
    # Success
    print(f"\n{'='*70}")
    print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    return True


def main():
    """Main entry point for the embedding pipeline runner."""
    while True:
        print_menu()
        
        try:
            choice = input("Please enter the number corresponding to your choice: ").strip()
            
            if choice == '1':
                # PDF Extraction
                if run_pdf_extraction():
                    response = input("\n[PROMPT] Run Embedding Generation next? (y/n): ").strip().lower()
                    if response == 'y':
                        if run_embedding_generation():
                            response = input("\n[PROMPT] Run Qdrant Upload next? (y/n): ").strip().lower()
                            if response == 'y':
                                run_qdrant_upload()
            
            elif choice == '2':
                # Embedding Generation
                if run_embedding_generation():
                    response = input("\n[PROMPT] Run Qdrant Upload next? (y/n): ").strip().lower()
                    if response == 'y':
                        run_qdrant_upload()
            
            elif choice == '3':
                # Qdrant Upload
                run_qdrant_upload()
            
            elif choice == '4':
                # Full Pipeline
                run_full_pipeline()
            
            elif choice == '5':
                # Exit
                print("\n[INFO] Exiting Embedding Pipeline Runner.")
                break
            
            else:
                print("[ERROR] Invalid choice. Please enter 1-5.")
        
        except KeyboardInterrupt:
            print("\n\n[INFO] Pipeline interrupted by user.")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
