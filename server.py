import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode

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

# Setup Flask
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = 'model/model.h5'
LABELS_PATH = 'model/labels.txt'

# Patch DepthwiseConv2D to ignore 'groups' param if it exists
class DepthwiseConv2DPatched(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove unsupported param
        super().__init__(*args, **kwargs)

get_custom_objects()['DepthwiseConv2D'] = DepthwiseConv2DPatched

# Model Loader Class
class ModelLoader:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.load_attempted = False

    def load_model(self):
        if self.load_attempted:
            return
        self.load_attempted = True

        # Load labels
        try:
            with open(LABELS_PATH, "r") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            logger.info(f"✅ Loaded {len(self.class_names)} classes")
        except Exception as e:
            logger.error(f"❌ Label loading failed: {str(e)}")
            return

        # Load model
        try:
            self.model = load_model(MODEL_PATH, compile=False)
            logger.info("✅ Model file loaded")

            # Optional: Fix input shape if needed
            expected_shape = (224, 224, 3)
            actual_shape = self.model.input_shape[1:]
            if actual_shape != expected_shape:
                logger.warning(f"⚠️ Input shape mismatch. Expected {expected_shape}, got {actual_shape}")
                new_input = Input(shape=expected_shape)
                output = self.model(new_input)
                self.model = Model(inputs=new_input, outputs=output)

            # Compile
            self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            # Warm up model
            self.model.predict(np.zeros((1, 224, 224, 3)))
            logger.info("✅ Model loaded and ready!")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
            self.model = None

# Instantiate the model loader
model_loader = ModelLoader()

# Health check route
@app.route('/health', methods=['GET'])
def health():
    model_loader.load_model()
    return jsonify({
        'model_loaded': model_loader.model is not None,
        'labels_loaded': len(model_loader.class_names) > 0,
        'status': 'ready' if model_loader.model else 'failed'
    })

# Prediction endpoint
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

# Run the server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
