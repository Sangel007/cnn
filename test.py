import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Function to preprocess the image
def preprocess_image(img_path, target_size=(64, 64)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
img_path=r'D:\project'
# Function to predict age from an image
def predict_age(model, img_path):
    img_array = preprocess_image(img_path)
    predicted_age = model.predict(img_array)
    return int(predicted_age[0][0])

# Function to display the image with a frame
def display_image_with_frame(img_path, predicted_age, frame_color='blue', frame_width=10):
    # Load and display the original image
    img = load_img(img_path)

    # Set up the figure and the axis
    fig, ax = plt.subplots(figsize=(5, 5))

    # Display the image
    ax.imshow(img)

    # Set frame by modifying the border aesthetics
    for spine in ax.spines.values():
        spine.set_edgecolor(frame_color)  # Set the frame color
        spine.set_linewidth(frame_width)  # Set the frame width

    # Add title (predicted age)
    plt.title(f"Predicted Age: {predicted_age}", fontsize=16)

    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Display the framed image
    plt.show()

# Load the trained model
model = load_model('age_detection_model.h5')

# Test with an example image (replace with the actual path to your test image)
test_img_path = 'test.jpg'
predicted_age = predict_age(model, test_img_path)

# Display the image with a frame
display_image_with_frame(test_img_path, predicted_age, frame_color='red', frame_width=8)
