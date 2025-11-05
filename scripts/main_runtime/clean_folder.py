"""
clean_folder.py
---------------
Utility function to safely clear all files and subfolders
inside a given directory. Used across the Medical_LMM project
to reset temp folders before new runs.
"""

import os
import shutil

def clean_folder(folder_path: str):
    """
    Deletes all files and subdirectories inside a folder,
    but keeps the folder itself.

    Args:
        folder_path (str): Absolute or relative path to the directory.
    """
    if not os.path.exists(folder_path):
        print(f"[clean_folder] Folder not found: {folder_path}")
        return

    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"[clean_folder] Could not delete {item_path}: {e}")
        print(f"[clean_folder] Cleared: {folder_path}")
    except Exception as e:
        print(f"[clean_folder] Error accessing {folder_path}: {e}")
