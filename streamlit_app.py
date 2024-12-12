import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import gdown

# Set the background and custom styles using HTML and CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&family=Open+Sans:wght@300&display=swap');
    
    .stApp {
        background-image: url('https://your-image-url.com');  # Use your background image URL here
        background-size: cover;
        font-family: 'Open Sans', sans-serif;
    }

    .title {
        font-family: 'Roboto', sans-serif;
        font-size: 50px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-top: 20px;
    }

    .result {
        font-size: 30px;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-box {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .image-box {
        text-align: center;
        margin-top: 20px;
    }

    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Weather Image Classifier</div>', unsafe_allow_html=True)
st.write("Upload an image to classify the weather condition.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))
    
    # Display the uploaded image
    st.markdown('<div class="image-box">', unsafe_allow_html=True)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Download model from Google Drive using gdown (if not already downloaded)
    model_url = 'https://drive.google.com/uc?export=download&id=14khsH-JbcQZ-OqBl2DxcEwhUsfKLyf1K'
    output = 'Best_Trained_Model.h5'
    gdown.download(model_url, output, quiet=False)

    # Load the trained model
    model = tf.keras.models.load_model(output)

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
    
    st.success('Prediction completed!')

    # Display result
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #4CAF50;'>Predicted Class: {predicted_class}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color: #FF9800;'>Confidence: {confidence:.2f}</h4>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Add a button to trigger the prediction again (optional)
    if st.button('Classify Another Image'):
        st.experimental_rerun()
