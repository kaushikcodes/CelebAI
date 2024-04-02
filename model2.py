import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
img_size = 64
batch_size = 32
import os

# Load and prepare the training data
train_dir = "celebrities_cropped_real"



train_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='categorical'  # Use 'categorical' for multi-class classification
)

class_names = train_dataset.class_names
print("Class names in training dataset:", class_names)
import json

# Save class names to a JSON file
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

validation_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='categorical'
)


# Updated model with Dropout
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.05),  # Dropout layer after pooling
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.05),  # Dropout layer after pooling
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),  # Dropout before the final dense layer
    layers.Dense(100, activation='softmax')  # 100 classes for 100 different celebrities
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=8
)

# Save the model for later use
model.save('celebrity_recognition_model2.h5')

print('training done')

model_path = 'celebrity_recognition_model2.h5'
model = load_model(model_path)
#class_names_path = 'class_names.json'

test_dir = "test_cropped"
img_size = 64  # Make sure this matches your model's expected input size
batch_size = 32  # Adjust based on your setup

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode=None,
    shuffle=False
)

predictions = model.predict(test_dataset)
predicted_indices = np.argmax(predictions, axis=1)

# Assuming test_dataset.file_paths is available and contains filenames in the order processed
filenames = test_dataset.file_paths
ids = [int(os.path.basename(name).split('.')[0]) for name in filenames]
expected_ids = set(range(0, 4977))  # Assuming IDs go from 0 to 4976

# Extract the IDs that you have predictions for
predicted_ids = set(ids)  # 'ids' being the list you previously extracted from filenames
# Find the IDs for which you don't have predictions
missing_ids = expected_ids - predicted_ids


predicted_celebrities = [class_names[i] for i in predicted_indices]
id_name_pairs = list(zip(ids, predicted_celebrities))
# For each missing ID, assign a random class from your list of class names
missing_predictions = {missing_id: np.random.choice(class_names) for missing_id in missing_ids}

# Now, combine your existing predictions with the randomly assigned ones for missing IDs
# Assuming 'id_name_pairs' contains your original (id, predicted_celebrity) tuples
all_predictions = list(id_name_pairs) + list(missing_predictions.items())

# Sort all predictions by ID to ensure they are in the correct order
all_predictions_sorted = sorted(all_predictions, key=lambda x: x[0])
#id_name_pairs_sorted = sorted(id_name_pairs, key=lambda x: x[0])
# Create and save DataFrame
predictions_df = pd.DataFrame(all_predictions_sorted, columns=['Id', 'Category'])
predictions_df.to_csv('test_predictions.csv', index=False)
