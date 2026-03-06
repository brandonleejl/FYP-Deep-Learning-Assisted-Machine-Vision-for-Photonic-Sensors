import os
import csv
import re
from PIL import Image
import numpy as np
import pytesseract

# --- Configuration ---
image_folder = r'C:\Users\brand\Downloads\FYP Code\FYP Dataset All Images'
output_csv = 'labels_color_masked.csv'

# Ensure the path matches your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_ph_and_create_csv():
    if not os.path.exists(image_folder):
        print(f"Error: Could not find the directory '{image_folder}'.")
        return

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'ph'])
        
        for file_name in os.listdir(image_folder):
            if file_name.upper().endswith('.PNG'):
                img_path = os.path.join(image_folder, file_name)
                
                try:
                    # Open the image in RGB mode
                    img = Image.open(img_path).convert('RGB')
                    
                    # Convert to a NumPy array for ultra-fast pixel filtering
                    img_array = np.array(img)
                    
                    # Separate the Red, Green, and Blue color channels
                    r_channel = img_array[:, :, 0]
                    g_channel = img_array[:, :, 1]
                    b_channel = img_array[:, :, 2]
                    
                    # COLOR MASK: Find pixels that are very Red, but NOT Green or Blue.
                    # This instantly ignores the glowing hydrogel and isolates your red text.
                    red_mask = (r_channel > 150) & (g_channel < 100) & (b_channel < 100)
                    
                    # Create a blank white canvas of the exact same size
                    processed_img_array = np.ones_like(img_array) * 255
                    
                    # Stamp the isolated red text onto the white canvas in solid Black (0, 0, 0)
                    processed_img_array[red_mask] = [0, 0, 0]
                    
                    # Convert back to an image for Tesseract to read
                    processed_img = Image.fromarray(processed_img_array)
                    
                    # OCR EXTRACTION
                    # psm 6 assumes a single block of text, which fits our new clean canvas perfectly
                    custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789.pH'
                    extracted_text = pytesseract.image_to_string(processed_img, config=custom_config)
                    
                    # Ensure it only captures the 3.0 to 8.0 increments
                    match = re.search(r'([3-8]\.\d|[3-8])', extracted_text)
                    ph_value = match.group(1) if match else 'NOT_FOUND'
                    
                    writer.writerow([file_name, ph_value])
                    print(f"Processed {file_name} -> Found pH: {ph_value}")
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    writer.writerow([file_name, 'ERROR'])
                
    print(f"\nDone! Please review '{output_csv}'.")

if __name__ == '__main__':
    extract_ph_and_create_csv()

# --- 2. Load and Preprocess Data ---
def load_data(images_dir, masks_dir):
    # FIX: Filter the list to ONLY include image files, ignoring .json
    all_files = sorted(os.listdir(images_dir))
    image_names = [f for f in all_files if f.upper().endswith(('.PNG', '.JPG', '.JPEG'))]
    
    images = []
    masks = []
    
    print(f"Found {len(image_names)} valid images. Loading into memory...")
    
    for img_name in image_names:
        img_path = os.path.join(images_dir, img_name)
        
        # Check for the matching mask. (Adjust the .PNG replacement if your masks use a different extension)
        mask_name = img_name.upper().replace('.JPG', '.PNG').replace('.JPEG', '.PNG')
        mask_path = os.path.join(masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue # Skip if there is no matching mask
            
        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            continue # Failsafe in case a file is corrupted
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        
        # Load and resize mask (Grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        
        images.append(img)
        masks.append(mask)

    # Normalize pixel values to be between 0 and 1
    X = np.array(images, dtype=np.float32) / 255.0
    # Masks need an extra dimension at the end: (224, 224) -> (224, 224, 1)
    y = np.expand_dims(np.array(masks, dtype=np.float32) / 255.0, axis=-1) 
    
    # Threshold masks to be strictly 0 or 1
    y = np.where(y > 0.5, 1.0, 0.0)
    
    return X, y 