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

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join('model', 'model.h5')
LABELS_PATH = os.path.join('model', 'labels.txt')

# Custom DepthwiseConv2D to handle version incompatibility
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

    def load_model(self):
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

        # Load model with proper input handling
        try:
            # Load the model
            self.model = load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D}
            )
            logger.info("✅ Model loaded successfully")

            # Fix multi-input issue by creating a new single-input model
            if len(self.model.inputs) > 1:
                logger.info("🔧 Converting multi-input model to single input...")
                new_input = Input(shape=(224, 224, 3))
                output = self.model([new_input, new_input])  # Duplicate input for both expected inputs
                self.model = Model(inputs=new_input, outputs=output)

            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Warm up the model
            logger.info("🔥 Warming up model...")
            self.model.predict(np.zeros((1, 224, 224, 3)))
            logger.info("✅ Model ready for predictions!")

        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
            self.model = None

# Initialize model loader
model_loader = ModelLoader()

@app.route('/health', methods=['GET'])
def health():
    model_loader.load_model()
    return jsonify({
        'model_loaded': model_loader.model is not None,
        'labels_loaded': len(model_loader.class_names) > 0,
        'status': 'ready' if model_loader.model else 'failed'
    })

@app.route('/predict', methods=['POST'])
def predict():
    model_loader.load_model()
    
    if not model_loader.model:
        return jsonify({
            "error": "Model not available",
            "details": "Server failed to load the AI model"
        }), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        img = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model_loader.model.predict(img_array)[0]
        return jsonify({
            "predictions": [
                {"className": model_loader.class_names[i], "probability": float(preds[i])}
                for i in range(len(model_loader.class_names))
            ],
            "predictedClass": model_loader.class_names[np.argmax(preds)],
            "confidence": float(np.max(preds))
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)