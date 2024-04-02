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

    # If no faces are detected, return None
    if len(faces) == 0:
        resized_img = cv2.resize(img, (224,224))
        return resized_img

    # Process only the first face found (1 face per image constraint)
    for (x, y, w, h) in faces[:1]:
        face_img = img[y:y+h, x:x+w]
        resized_face_img = cv2.resize(face_img, (224, 224))  # Resize to a fixed size
        return resized_face_img

# Load the CSV file
df = pd.read_csv('train.csv')

# Define the base directory for images
base_dir = 'test_kag'
img_dir = 'train'

# Ensure the base directory exists
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Iterate over the DataFrame rows
for index, row in df.iterrows():
    img_path = os.path.join(img_dir, row['File Name'])
    celebrity_name = row['Category']
    
    # Define the target directory based on the celebrity name
    target_dir = os.path.join(base_dir, celebrity_name)
    
    # Check if the target directory exists, create if not
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Define the target path for the processed image
    target_path = os.path.join(target_dir, row['File Name'])
    
    # Process the image to extract and resize the face
    processed_img = extract_features(img_path)
    if processed_img is not None:
        # Save the processed image to the target directory
        cv2.imwrite(target_path, processed_img)
        print(f"Processed and saved {img_path} to {target_path}")
    else:
        print(f"No face detected in {img_path}, skipped.")
