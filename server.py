import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import numpy as np
from PIL import Image
import io
import logging
import threading

# Setup Flask
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths - ensure these match your Render.com file structure
MODEL_PATH = os.path.join('model', 'model.h5')
LABELS_PATH = os.path.join('model', 'labels.txt')

# Custom DepthwiseConv2D to handle version compatibility
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove problematic parameter
        super().__init__(*args, **kwargs)

# Register our custom layer
get_custom_objects()['DepthwiseConv2D'] = PatchedDepthwiseConv2D

class ModelLoader:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.load_attempted = False
        self.lock = threading.Lock()

    def load_model(self):
        with self.lock:
            if self.load_attempted:
                return
            self.load_attempted = True

            # Load labels first
            try:
                with open(LABELS_PATH, "r") as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
                logger.info(f"✅ Loaded {len(self.class_names)} classes")
            except Exception as e:
                logger.error(f"❌ Label loading failed: {str(e)}")
                return

            # Load model with multiple fallback strategies
            try:
                try:
                    # First attempt: regular load
                    self.model = load_model(
                        MODEL_PATH,
                        compile=False,
                        custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D}
                    )
                except Exception as e:
                    logger.warning(f"First load attempt failed, trying alternative: {str(e)}")
                    # Fallback: load weights only
                    from tensorflow.keras.models import model_from_json
                    with open(MODEL_PATH.replace('.h5', '.json'), 'r') as json_file:
                        self.model = model_from_json(json_file.read())
                    self.model.load_weights(MODEL_PATH)

                # Verify and fix input shape if needed
                if len(self.model.inputs) > 1:
                    logger.info("🔧 Fixing multi-input model...")
                    new_input = Input(shape=(224, 224, 3), name='fixed_input')
                    output = self.model(new_input)
                    self.model = Model(inputs=new_input, outputs=output)

                # Compile model
                self.model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Warm up the model
                logger.info("🔥 Warming up model...")
                self.model.predict(np.zeros((1, 224, 224, 3)))
                logger.info("✅ Model loaded and ready!")

            except Exception as e:
                logger.error(f"❌ Model loading failed: {str(e)}")
                self.model = None

# Initialize model loader
model_loader = ModelLoader()

# Start loading the model in background
threading.Thread(target=model_loader.load_model, daemon=True).start()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'model_loaded': model_loader.model is not None,
        'labels_loaded': len(model_loader.class_names) > 0,
        'status': 'ready' if model_loader.model else 'initializing',
        'endpoints': {
            'POST /predict': 'Image classification endpoint',
            'GET /health': 'Service health check'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model_loader.model is None:
        return jsonify({
            "error": "Model not ready",
            "details": "Server is still initializing the AI model"
        }), 503

    if 'image' not in request.files:
        return jsonify({
            "error": "No image provided",
            "details": "Please include an image file in your request"
        }), 400

    try:
        # Process image
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        img = Image.open(io.BytesIO(file.read()))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model_loader.model.predict(img_array)[0]
        
        return jsonify({
            "predictions": [
                {
                    "className": model_loader.class_names[i],
                    "probability": float(predictions[i])
                }
                for i in range(len(model_loader.class_names))
            ],
            "predictedClass": model_loader.class_names[np.argmax(predictions)],
            "confidence": float(np.max(predictions))
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Use PORT from environment for Render.com
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)