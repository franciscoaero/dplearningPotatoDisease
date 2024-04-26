import os
import json
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Initialize global variables
def init():
    global model
    global class_indices
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'plant_disease_model_v1.keras')
    model = tf.keras.models.load_model(model_path)

    class_indices_path = 'class_indices.json'
    with open(class_indices_path) as f:
        class_indices = json.load(f)

# Function to predict the class of an image
def predict_image_class(image_data):
    image_treat = Image.open(io.BytesIO(image_data))
    img_batch = np.expand_dims(image_treat, 0)
    predictions = model.predict(img_batch)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Run the prediction
def run(raw_data):
    try:
        # Convert the JSON string to bytes
        image_data = json.loads(raw_data)['data']
        image_data = base64.b64decode(image_data)
        predicted_class_name = predict_image_class(image_data)
        return json.dumps({"Predicted Class Name": predicted_class_name})
    except Exception as e:
        error = str(e)
        return error
