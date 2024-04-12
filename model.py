import os
import numpy as np
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model

img_size = 64
batch_size = 32
train_function = "celebrities_cropped"

train_dataset = image_dataset_from_directory(
    train_function,
    validation_split=0.05,
    subset="training",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='categorical'
)

validation_dataset = image_dataset_from_directory(
    train_function,
    validation_split=0.05,
    subset="validation",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train_dataset.class_names
print("Class names in training dataset:", class_names)

fptr = open('class_names.json', 'w')
fptr.write(json.dump(class_names))
fptr.close()

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.05), #next cnn layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), #next cnn layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), #next cnn layer
    layers.Dropout(0.05),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),  
    layers.Dense(100, activation='softmax') #because 100 classes
])

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=8)
model.save('celebrity_recognition_model3.h5')

print('training done')

model_path = 'celebrity_recognition_model3.h5'
model = load_model(model_path)

testing_dataset = "test_cropped"
img_size = 64  
batch_size = 32  

test_dataset = image_dataset_from_directory(
    testing_dataset,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode=None,
    shuffle=False
)

predictions = model.predict(test_dataset)
predicted_indices = np.argmax(predictions, axis=1)

filenames = test_dataset.file_paths
id_list = []
for i in filenames:
    id_ = int(os.path.basename(i).split('.')[0])
    id_list.append(id_)

expected_ids = set(range(0, 4977))
predicted_ids = set(id_list)  
missing_ids = expected_ids - predicted_ids

predicted_celebrities = []
for i in range(len(predicted_indices)):
    predicted_celebrities.append(class_names[predicted_indices[i]])

id_name_pairs = []
for i in range(len(id_list)):
    pair = (id_list[i], predicted_celebrities[i])
    id_name_pairs.append(pair)

missing_predictions = {}
for missing_id in missing_ids:
    random_class = np.random.choice(class_names)
    missing_predictions[missing_id] = random_class

all_predictions = []
for pair in id_name_pairs:
    all_predictions.append(pair)
for item in missing_predictions.items():
    all_predictions.append(item)

all_predictions_sorted = sorted(all_predictions, key=lambda x: x[0])

predictions_df = pd.DataFrame(all_predictions_sorted, columns=['Id', 'Category'])
predictions_df.to_csv('last_test_predictions.csv', index=False)