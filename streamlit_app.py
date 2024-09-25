import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import os 
# Labels for the skin conditions
labels = [ 'Benign Keratosis-like Lesions (bkl)', 'Melanocytic Nevi (nv)', 
          'Dermatofibroma (df)', 'Melanoma (mel)', 'Vascular Lesions (vasc)', 
          'Basal Cell Carcinoma (bcc)', 'Actinic Keratoses and Intraepithelial Carcinoma (akiec)']


# Load the trained model

url = 'https://drive.google.com/uc?id=1yLlCYR9b6IPCpGtwIP2SBaT0dcOoX2Wh'
output = 'best_model.keras'

if not os.path.exists("best_model.keras"):
    with st.spinner("Downloading model, please wait..."):
        gdown.download(url, output, quiet=False)

model = tf.keras.models.load_model('best_model.keras')
input_shape = model.input_shape[1:3]  # Input shape of the model

# Define Streamlit app
st.title("Skin Condition Detector")
st.write("Upload an image of a skin lesion to get the prediction:")
st.write("Detect all these labels)
st.write("Melanocytic Nevi (nv)")
st.write("Dermatofibroma (df)")
st.write("Vascular Lesions (vasc)")
st.write("Basal Cell Carcinoma (bcc)")
st.write("Actinic Keratoses and Intraepithelial Carcinoma (akiec)")



# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    try:
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
            st.write('Prediction:')
            st.header(predicted_label)
    except:
        st.warning("Please upload a valid image file.")


