import fitz
import os

def extract_image(pdf_path):
    try:
        doc = fitz.open(pdf_path)

        output_dir = "preprocessor/res/extracted_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Extracting images to the '{output_dir}' folder...")

        image_count = 0
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            image_list = page.get_images(full=True)
            
            if not image_list:
                continue # Skip pages with no images

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                
                # code for extracting image dimensions
                try:
                    pix = fitz.Pixmap(doc, xref)
                    width = pix.width
                    height = pix.height
                    print(f"Dimensions: {width}x{height}")
                    pix = None
                except Exception as e:
                    print(f"Error processing image with XREF {xref}: {e}")


                # Extract the raw image data using the xref
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                print(image_filename)
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                image_count += 1

        print(f"Done! Found and extracted {image_count} images.")

        doc.close()

    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")