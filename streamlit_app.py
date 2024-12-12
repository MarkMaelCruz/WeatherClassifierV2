import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import gdown

# Download model from Google Drive using gdown
model_url = 'https://drive.google.com/uc?export=download&id=14khsH-JbcQZ-OqBl2DxcEwhUsfKLyf1K'  # Updated model URL
output = 'Best_Trained_Model.h5'
gdown.download(model_url, output, quiet=False)

# Load the best-trained model
model = tf.keras.models.load_model(output)

# Define class labels
class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

st.title('Weather Image Classifier')
st.write("Upload an image to classify the weather condition.")

uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
