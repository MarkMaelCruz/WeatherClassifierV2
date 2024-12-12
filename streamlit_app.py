import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import base64

# Convert image to base64 for Streamlit background
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Replace with your image path
base64_image = image_to_base64('Wallpaper11.jpg')  # Make sure 'Wallpaper11.jpg' is in the same directory

# Set background using the base64 image
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&family=Open+Sans:wght@300&display=swap');
    
    .stApp {{
        background-image: url('data:image/jpeg;base64,{base64_image}');
        background-size: cover;
        font-family: 'Open Sans', sans-serif;
    }}

    .title {{
        font-family: 'Roboto', sans-serif;
        font-size: 50px;
        font-weight: bold;
        color: #000000;
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.7);  /* White background with transparency */
        border-radius: 10px;
    }}

    .result {{
        font-size: 30px;
        color: #000000;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.8);  /* White background with transparency */
        border-radius: 10px;
    }}

    .upload-section {{
        padding: 20px;
        margin-top: 30px;
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.8); /* Light white background */
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }}

    .success-msg {{
        font-size: 20px;
        color: #000000;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.8);  /* White background with transparency */
        border-radius: 10px;
    }}

    .classify-btn {{
        margin-top: 10px;  /* Move button down by 10px */
    }}
    </style>
    """, unsafe_allow_html=True)

# Title (Black color)
st.markdown('<div class="title">Weather Image Classifier</div>', unsafe_allow_html=True)
st.write("Upload an image to classify the weather condition.")

# Image upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the trained model
    model = tf.keras.models.load_model('Best_Trained_Model.h5')

    # Define class labels
    class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

    # Preprocess the image
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the class
    with st.spinner('Classifying your image...'):
        prediction = model.predict(image_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
    
    # Display success message
    st.markdown('<div class="success-msg">Prediction completed!</div>', unsafe_allow_html=True)

    # Display result
    st.markdown(f"<div class='result'>Predicted Class: {predicted_class}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result'>Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)

    # Add a button to trigger the prediction again with adjusted margin
    st.markdown('<div class="classify-btn">', unsafe_allow_html=True)
    if st.button('Classify Another Image'):
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close upload-section div
