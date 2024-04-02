import os
import cv2

# Define the base directory
base_dir = 'celebrities_cropped'

# Iterate through each subdirectory in the base directory
for celebrity_name in os.listdir(base_dir):
    celebrity_dir = os.path.join(base_dir, celebrity_name)
    
    # Check if the path is indeed a directory
    if os.path.isdir(celebrity_dir):
        # Iterate through each file in the celebrity's directory
        for image_name in os.listdir(celebrity_dir):
            # Construct the full path to the image file
            image_path = os.path.join(celebrity_dir, image_name)
            
            # Ensure the file is an image (e.g., ends with .jpg)
            if image_path.lower().endswith('.jpg'):
                # Read the image using OpenCV
                img = cv2.imread(image_path)
                
                # Check if the image was successfully loaded
                if img is not None:
                    # Get the image dimensions (height, width)
                    height, width = img.shape[:2]
                    if(height != 64 or width != 64):
                        resized_img = cv2.resize(img, (64, 64))
                        cv2.imwrite(image_path, resized_img)
                    print(f"{image_name} (in {celebrity_name}): {width}x{height}")
                else:
                    print(f"Failed to load image: {image_path}")
