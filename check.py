import pandas as pd
import os
import shutil
import cv2

def extract_features(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None  # Early return if image is not loaded

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(224, 224),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If no faces are detected, resize and return the whole image
    if len(faces) == 0:
        resized_img = cv2.resize(img, (224, 224))
        return resized_img

    # Process only the first face found (1 face per image constraint)
    for (x, y, w, h) in faces[:1]:
        face_img = img[y:y+h, x:x+w]
        resized_face_img = cv2.resize(face_img, (224, 224))  # Resize to a fixed size
        return resized_face_img

# Define the base and target directories for images
img_dir = 'test_kag'
target_dir = 'test_transfer'

# Ensure the target directory exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Iterate through each file in the image directory
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    
    # Skip if the file is not an image
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue
    
    # Process the image
    processed_img = extract_features(img_path)
    if processed_img is not None:
        # Define the target path for the processed image
        target_path = os.path.join(target_dir, img_name)
        
        # Save the processed image to the target directory
        cv2.imwrite(target_path, processed_img)
        print(f"Processed and saved {img_name} to {target_path}")
    else:
        print(f"No face detected in {img_name}, skipped.")

