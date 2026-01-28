import os
import cv2
from tqdm import tqdm

# Path to your unzipped micrographs
input_dir = 'data/micrographs'
output_dir = 'data/processed_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_images():
    print("✂️ Microstructure cropping start ho rahi hai...")
    patch_size = 64 # IHI project ke liye standard size
    count = 0
    
    # Loop through micrographs folders
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith(('.jpg', '.png', '.tif')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None: continue
                
                # Image ko chote squares mein kaatna
                h, w = img.shape
                for i in range(0, h - patch_size, patch_size):
                    for j in range(0, w - patch_size, patch_size):
                        patch = img[i:i+patch_size, j:j+patch_size]
                        cv2.imwrite(f"{output_dir}/patch_{count}.jpg", patch)
                        count += 1
                        
    print(f"✅ Success! {count} patches ban gaye hain '{output_dir}' folder mein.")

if __name__ == "__main__":
    process_images()