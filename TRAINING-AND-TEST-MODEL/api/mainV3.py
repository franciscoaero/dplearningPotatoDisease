import os
import io
from io import BytesIO
import json
import base64
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_model_v1.keras"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Predict the Class of an Image
def predict_image_class(model, image_data, class_indices):
    image_treat = Image.open(io.BytesIO(image_data))
    img_batch = np.expand_dims(image_treat, 0)
    predictions = model.predict(img_batch)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title('Potato Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr = byte_arr.getvalue()

    # Convert image to base64
    image_base64 = base64.b64encode(byte_arr).decode('utf-8')
    json_request = json.dumps({'data': image_base64})

    # Save this JSON as a file
    json_file_path = 'sample-request.json'  # Specify your desired path for the JSON file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_request)

    image_data = json.loads(json_request)['data']
    image_data = base64.b64decode(image_data)

    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, image_data, class_indices)
            st.success(f'Prediction: {str(prediction)}')