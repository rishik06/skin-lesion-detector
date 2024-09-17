import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# Labels for the skin conditions
labels = ['Benign Keratosis-like Lesions (bkl)', 'Melanocytic Nevi (nv)', 
          'Dermatofibroma (df)', 'Melanoma (mel)', 'Vascular Lesions (vasc)', 
          'Basal Cell Carcinoma (bcc)', 'Actinic Keratoses and Intraepithelial Carcinoma (akiec)']

# Load the trained model
model = tf.keras.models.load_model('best_model.keras')
input_shape = model.input_shape[1:3]  # Input shape of the model

# Define Streamlit app
st.title("Skin Condition Detector")
st.write("Upload an image of a skin lesion to get the prediction:")
st.write(labels)
# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    with st.spinner("classifying"):    
        # Preprocess the image
        img = img.resize(input_shape)  # Resize the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Make predictions
        predictions = model.predict(img_array)
        
        # Get the index of the maximum prediction score
        predicted_label_index = np.argmax(predictions[0])
        predicted_label = labels[predicted_label_index]
        
        # Display the prediction result
        st.header(f'Prediction: **{predicted_label}**')


