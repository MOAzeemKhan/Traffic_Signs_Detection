import tensorflow as tf
import cv2
import numpy as np

# Load the pre-trained GTSRB model (.h5 file)
model = tf.keras.models.load_model('my_model2.h5')

# Function to preprocess the image before passing it to the GTSRB model
def preprocess_image(image):
    # Resize the image to match the input size expected by the GTSRB model (32x32 or as required by your model)
    resized_image = cv2.resize(image, (30, 30))  # Modify size based on your model input
    # Normalize the image
    normalized_image = resized_image.astype('float32') / 255.0
    # Expand dimensions to match the input shape (1, 32, 32, 3) for batch processing
    return np.expand_dims(normalized_image, axis=0)

# Class names corresponding to speed limits (example mapping, adjust based on your model training)
class_names = {
    0: '20 km/h',
    1: '30 km/h',
    2: '50 km/h',
    3: '60 km/h',
    4: '70 km/h',
    5: '80 km/h',
    6: 'End of Speed Limit 80 km/h',
    7: '100 km/h',
    8: '120 km/h'
}

# Function to classify the speed limit from the image of the bounding box
def classify_speed_limit(cropped_image):
    # Preprocess the cropped image
    preprocessed_image = preprocess_image(cropped_image)

    # Predict the class using the model
    predictions = model.predict(preprocessed_image)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    # Map the predicted class to the corresponding speed limit
    predicted_speed_limit = class_names.get(predicted_class, 'Unknown Speed Limit')

    return predicted_speed_limit
