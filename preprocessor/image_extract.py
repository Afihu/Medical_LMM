import fitz 
import os
import glob
import numpy as np  

# Function of interest
def extract_image(pdf_path):
    image_data = []
    
    pdf_name = os.path.basename(pdf_path)
    print(f"\n--- Processing: {pdf_name} ---")

    try:
        doc = fitz.open(pdf_path)
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
        return image_data
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return image_data

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        
        # Get the list of images (xref is the first element)
        image_list = page.get_images(full=True)
        
        if not image_list:
            continue

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            
            try:
                # Get the image as a Pixmap (in-memory bitmap)
                pix = fitz.Pixmap(doc, xref)

                # Convert to RGB 
                if pix.n > 3:  # Checks if it has more than 3 color components
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                elif pix.colorspace.name != "DeviceRGB":
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_array = np.frombuffer(
                    pix.samples,
                    dtype=np.uint8
                ).reshape(pix.height, pix.width, 3)

                print(f"  Extracted image {img_index+1} from page {page_num+1} (Shape: {image_array.shape})")

                image_data.append(image_array)

            except Exception as e:
                print(f"  Error processing image xref {xref} on page {page_num+1}: {e}")
            
            pix = None

    doc.close()
    
    print(f"--- Finished processing {pdf_name}. Total extracted: {len(image_data)} images. ---")
    return image_data

# Just in case
def extract_images(pdf_directory): #returns an rgb array
    
    all_images_data = []
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

                    image_array = np.frombuffer(
                        pix.samples,
                        dtype=np.uint8
                    ).reshape(pix.height, pix.width, 3)

                    all_images_data.append(image_array)
                    
                    print(f"  Extracted image {img_index+1} from page {page_num+1} (Shape: {image_array.shape})")

                except Exception as e:
                    print(f"  Error processing image xref {xref} on page {page_num+1}: {e}")
                
                pix = None

        doc.close()

    print(f"\n--- Extraction Complete ---")
    print(f"Successfully extracted {len(all_images_data)} images")
    return all_images_data
