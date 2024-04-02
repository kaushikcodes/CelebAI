import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = Dense(512, activation='relu')(x)
# Add a logistic layer for 100 classes (assuming you have 100 celebrity classes)
predictions = Dense(100, activation='softmax')(x)
img_size=224
batch_size=32
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


train_generator = image_dataset_from_directory(
    'celebrities_cropped_real',  # This should be the path to the training data
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='categorical'  # Use 'categorical' for multi-class classification
    )  # Set as training data

validation_generator = image_dataset_from_directory(
    'celebrities_cropped_real',  # This should be the path to the training data
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='categorical')  # Set as validation data

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator)
# Unfreeze the last few layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile the model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Use a lower learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training
history_fine = model.fit(
    train_generator,
    epochs=8,  # Number of epochs for fine-tuning
    validation_data=validation_generator)


#test_datagen = ImageDataGenerator(rescale=1./255)

test_dataset = image_dataset_from_directory(
    'test_cropped',  # This should be the path to the testing data
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # Because we do not have labels in the test set
    shuffle=False)  # Important for matching predictions to file names/IDs


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
