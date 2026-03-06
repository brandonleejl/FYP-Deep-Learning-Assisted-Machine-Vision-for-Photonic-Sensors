import os
import re
from PIL import Image
import numpy as np
import pytesseract

# --- Configuration ---
image_folder = r'C:\Users\brand\Downloads\FYP Code\FYP Dataset All Images'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def validate_ocr_failures(max_samples=5):
    if not os.path.exists(image_folder):
        print(f"Error: Could not find the directory '{image_folder}'.")
        return

    failures_found = 0
    print("Scanning for OCR failures to debug...\n")

    for file_name in os.listdir(image_folder):
        if file_name.upper().endswith('.PNG'):
            img_path = os.path.join(image_folder, file_name)
            
            try:
                # 1. Apply the exact same Color Masking logic
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                r_channel = img_array[:, :, 0]
                g_channel = img_array[:, :, 1]
                b_channel = img_array[:, :, 2]
                
                # The color threshold we are testing
                red_mask = (r_channel > 150) & (g_channel < 100) & (b_channel < 100)
                
                processed_img_array = np.ones_like(img_array) * 255
                processed_img_array[red_mask] = [0, 0, 0]
                processed_img = Image.fromarray(processed_img_array)
                
                # 2. Run Tesseract
                custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789.pH'
                extracted_text = pytesseract.image_to_string(processed_img, config=custom_config)
                
                match = re.search(r'([3-8]\.\d|[3-8])', extracted_text)
                
                # 3. If it fails, show us why
                if not match:
                    failures_found += 1
                    raw_text = extracted_text.strip().replace('\n', ' ')
                    print(f"[{failures_found}/{max_samples}] Failed on {file_name}.")
                    print(f"   -> What Tesseract thought it read: '{raw_text}'")
                    
                    # Save the image so you can look at it
                    debug_filename = f"debug_{file_name}"
                    processed_img.save(debug_filename)
                    print(f"   -> Saved visual output to: {debug_filename}\n")
                    
                    # Automatically open the image on your computer
                    processed_img.show()
                    
                    if failures_found >= max_samples:
                        print("Reached maximum sample limit. Stopping scan.")
                        break
                        
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == '__main__':
    validate_ocr_failures()