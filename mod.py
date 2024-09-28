import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import os

# 1. Load and Preprocess the Data
import os
import numpy as np
import tensorflow as tf

def load_data(data_directory, target_size=(64, 64)):
    images = []
    ages = []
    
    # Iterate through each folder in the dataset directory
    for folder_name in os.listdir(data_directory):
        folder_path = os.path.join(data_directory, folder_name)
        
        if os.path.isdir(folder_path):  # Check if it is a directory
            for img_name in os.listdir(folder_path):
                try:
                    # Extract age from the filename, which is formatted as "age_gender_race_date.jpg"
                    age = int(img_name.split('_')[0])
                    img_path = os.path.join(folder_path, img_name)
                    
                    # Load and resize the image
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    
                    # Append the image and the age label to the respective lists
                    images.append(img)
                    ages.append(age)
                except Exception as e:
                    print(f"Error loading image {img_name} in folder {folder_name}: {e}")

    print(f"Total loaded images: {len(images)}")
    print(f"Total loaded ages: {len(ages)}")

    images = np.array(images)
    ages = np.array(ages)
    
    return images, ages

# Set data directory path (replace with your dataset path)
data_directory =r'D:\project\dataset'  # Replace with the path to your dataset folder
images, ages = load_data(data_directory)

# Check the shapes of the loaded data
print(f"Images shape: {images.shape}")
print(f"Ages shape: {ages.shape}")

# Normalize pixel values to range [0, 1]
images = images / 255.0

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

train_images, val_images, train_ages, val_ages = train_test_split(
    images, ages, test_size=0.2, random_state=42
)

# 2. Data Augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,  # Randomly flip images horizontally
    rotation_range=20,     # Rotate images up to 20 degrees
    zoom_range=0.15,       # Random zoom
    width_shift_range=0.2, # Horizontal shift
    height_shift_range=0.2 # Vertical shift
)

# 3. Define the Model
def create_age_detection_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Single output for age regression
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

model = create_age_detection_model(input_shape=(64, 64, 3))
model.summary()

# 4. Training the Model
# Use the augmented data generator for training
train_generator = datagen.flow(train_images, train_ages, batch_size=32)
val_generator = ImageDataGenerator().flow(val_images, val_ages, batch_size=32)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # Set number of epochs
    steps_per_epoch=len(train_images) // 32,  # Number of steps per epoch
    validation_steps=len(val_images) // 32
)

# 5. Save the Model
model.save('age_detection_model.h5')

# 6. Evaluate the Model
test_loss, test_mae = model.evaluate(val_generator)
print(f"Test MAE: {test_mae}")

images, ages = load_data(data_directory)
print(f"Images shape: {images.shape}")
print(f"First image shape: {images[0].shape if len(images) > 0 else 'No images loaded'}")
print(f"Ages shape: {ages.shape}")
print(f"First few ages: {ages[:5] if len(ages) > 0 else 'No ages loaded'}")
