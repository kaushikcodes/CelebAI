import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tqdm import tqdm

    
dir_path = r"celebrities_cropped"
images = []
nums = []
img_size = 64
for i, name in tqdm(enumerate(os.listdir(dir_path))):
    folder_path = os.path.join(dir_path, name)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img_array = cv2.imread(img_path)
        if img_array is None:
            print(f"Warning: Could not load image at {img_path}. Skipping...")
            continue
        img_array = cv2.resize(img_array, (img_size, img_size))
        img_array = img_array[:, :, ::-1] / 255.0
        images.append(img_array)
        nums.append(i)
images = np.array(images, dtype = 'float32').reshape(-1, img_size, img_size, 3)
nums = np.array(nums, dtype = 'float32')
images.shape, nums.shape


class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def call(self, vec1, vec2):
        return tf.square(vec1 - vec2)
    
class SiameseNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential([
            layers.Conv2D(32, kernel_size = (3, 3), strides = 1, padding = 'same', activation = 'relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size = (2, 2), strides = 1),
            
            layers.Conv2D(32, kernel_size = (3, 3), strides = 1, padding = 'same', activation = 'relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size = (2, 2), strides = 1),
          
            layers.Conv2D(32, kernel_size = (3, 3), strides = 1, padding = 'same', activation = 'relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size = (2, 2), strides = 1),
            
            layers.Flatten(),
            layers.Dense(128, activation = 'relu'),
            layers.BatchNormalization(),
            layers.Dense(16)
        ])
        self.get_distance = DistanceLayer()
        self.output_layer = layers.Dense(1, activation = 'sigmoid')
        
    def call(self, args):
        x1, x2 = args
        embedding1, embedding2 = self.encoder(x1), self.encoder(x2)
        distance = self.get_distance(embedding1, embedding2)
        out = self.output_layer(distance)
        return out



X1 = []
X2 = []
y = []
same = 0
not_same = 0
for i in tqdm(range(len(images) - 1)):
        X1.append(images[i])
        X2.append(images[i + 1])
        y.append(np.float32(1.0))
        
        X1.append(images[i])
        X2.append(images[len(images) - i - 1])
        y.append(np.float32(0.0))
        
X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)
X1.shape, X2.shape, y.shape, y.sum()


from sklearn.model_selection import KFold

# Assuming X1, X2, and y are already prepared and are numpy arrays
num_folds = 3  # Number of folds you want to use
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists to store results of each fold
fold_no = 1
loss_per_fold = []
accuracy_per_fold = []

# KFold Cross Validation model evaluation
for train, test in kf.split(X1, y):
    # Clear model, and create it
    model = SiameseNetwork()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Generate a print
    print(f'Training for fold {fold_no} ...')
    
    # Fit data to model
    history = model.fit([X1[train], X2[train]], y[train], epochs=2)
    
    # Generate generalization metrics
    scores = model.evaluate([X1[test], X2[test]], y[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    loss_per_fold.append(scores[0])
    accuracy_per_fold.append(scores[1] * 100)
    
    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(accuracy_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {accuracy_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(accuracy_per_fold)} (+- {np.std(accuracy_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
