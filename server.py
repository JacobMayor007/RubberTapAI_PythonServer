from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D 
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import io
import os
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Fix for DepthwiseConv2D 'groups' error
def depthwise_conv2d_without_groups(**kwargs):
    kwargs.pop('groups', None)
    return DepthwiseConv2D(**kwargs)

# Model loading with custom handler
MODEL_PATH = os.path.join('model', 'model.h5')
LABELS_PATH = os.path.join('model', 'labels.txt')

try:
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={'DepthwiseConv2D': depthwise_conv2d_without_groups}
    )
    model.compile(optimizer=Adam(), 
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Model loading failed: {str(e)}")
    model = None

# Rest of your code remains the same...