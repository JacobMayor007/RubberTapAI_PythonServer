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
import threading

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom DepthwiseConv2D loader to handle 'groups' parameter
def depthwise_conv2d_without_groups(**kwargs):
    kwargs.pop('groups', None)  # Remove 'groups' if present
    return DepthwiseConv2D(**kwargs)

# Configuration
MODEL_PATH = os.path.join('model', 'model.h5')
LABELS_PATH = os.path.join('model', 'labels.txt')

# Global variables for frame processing
latest_frame = None
frame_lock = threading.Lock()
processing_enabled = False
analysis_results = {}

# Initialize model
try:
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={'DepthwiseConv2D': depthwise_conv2d_without_groups}
    )
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    logger.info("✅ Model loaded successfully!")
    
    # Warm up the model
    dummy_input = np.zeros((1, 224, 224, 3))
    model.predict(dummy_input)
except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")
    model = None

# Load class labels
try:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    logger.info(f"✅ Loaded {len(class_names)} classes")
except Exception as e:
    logger.error(f"❌ Label loading failed: {str(e)}")
    class_names = []

def preprocess_image(image_file):
    """Process uploaded image for prediction"""
    try:
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype("float32") / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    if model is None or not class_names:
        return jsonify({"error": "Model not ready"}), 503
        
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        if not file.content_type.startswith('image/'):
            return jsonify({"error": "File is not an image"}), 400
            
        img_array = preprocess_image(file)
        predictions = model.predict(img_array)[0]
        
        results = {
            "predictions": [
                {
                    "className": class_names[i],
                    "probability": float(predictions[i])
                }
                for i in range(len(class_names))
            ],
            "predictedClass": class_names[np.argmax(predictions)],
            "confidence": float(np.max(predictions))
        }
        
        logger.info(f"Prediction successful: {results['predictedClass']}")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Service health endpoint"""
    status = {
        'model_loaded': model is not None,
        'labels_loaded': len(class_names) > 0,
        'status': 'ready' if (model is not None and class_names) else 'not ready'
    }
    return jsonify(status)

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Server is running',
        'endpoints': {
            '/predict': 'POST image for classification',
            '/health': 'GET service health status'
        }
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, threaded=True)