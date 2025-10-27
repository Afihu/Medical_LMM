import fitz 
import os
import glob
import numpy as np  

def extract_images_as_rgb_arrays(pdf_directory):
    
    all_images_data = []
    
    # Find all PDF files
    pdf_paths = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    if not pdf_paths:
        print(f"Error: No PDF files found in the directory '{pdf_directory}'.")
        return all_images_data

    print(f"Found {len(pdf_paths)} PDF file(s). Starting extraction...")

    # Iterate
    for pdf_path in pdf_paths:
        print(f"\n--- Processing: {os.path.basename(pdf_path)} ---")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  Error opening {pdf_path}: {e}")
            continue  

        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            image_list = page.get_images(full=True)
            
            if not image_list:
                continue  # Skip pages with no images

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                
                try:
                    # Get the image as a Pixmap
                    pix = fitz.Pixmap(doc, xref)

                    # Convert to RGB if it's not
                    if pix.colorspace.name != "DeviceRGB" or pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    # Convert the Pixmap's samples into a NumPy array
                    image_array = np.frombuffer(
                        pix.samples,
                        dtype=np.uint8
                    ).reshape(pix.height, pix.width, 3)

                    # Add the RGB array to master list
                    all_images_data.append(image_array)
                    
                    print(f"  Extracted image {img_index+1} from page {page_num+1} (Shape: {image_array.shape})")

                except Exception as e:
                    print(f"  Error processing image xref {xref} on page {page_num+1}: {e}")
                
                pix = None

        doc.close()

    print(f"\n--- Extraction Complete ---")
    print(f"Successfully extracted {len(all_images_data)} images into memory.")
    return all_images_data


if __name__ == "__main__":
    pdf_folder_path = "source_materials" 

    if not os.path.isdir(pdf_folder_path):
        print(f"Directory not found: '{pdf_folder_path}'")
        print("Please update the 'pdf_folder_path' variable in the script.")
    else:
        # extraction
        list_of_rgb_arrays = extract_images_as_rgb_arrays(pdf_folder_path)

        # Debug
        if list_of_rgb_arrays:
            # Print info about the first image as a demo
            print(f"\nTotal images in list: {len(list_of_rgb_arrays)}")
            print(f"Type of first item: {type(list_of_rgb_arrays[0])}")
            print(f"Shape of first image array: {list_of_rgb_arrays[0].shape}")