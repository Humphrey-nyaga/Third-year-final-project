import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')

#print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/retinopathy_model.h5'

# Load your own trained model
model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')


def model_predict(img, model):
#resize image
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        preds = model_predict(img, model)
        print(preds)
        # Process highest max result from int64 to float
        pred_proba = "{:.3f}".format(np.amax(preds)) 
        print(pred_proba) 
          # Max probability
        pred_class = np.argmax(preds) #Find the index with highest prediction from array
        value= {0:"Mild", 1:"Moderate", 2:"NO_DR", 3:"Proliferate", 4:"Severe"}
        result = value[pred_class] # Convert to string
        result = result.replace('_', ' ').capitalize()
        
        
        # Serialize the result and pass as a JSON object
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
