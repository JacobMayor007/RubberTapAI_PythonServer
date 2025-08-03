import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import io
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = 'model/model.h5'
LABELS_PATH = 'model/labels.txt'

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

        # Load and fix model
        try:
            # First attempt regular load
            try:
                self.model = load_model(MODEL_PATH, compile=False)
            except Exception as e:
                logger.warning(f"Standard load failed, attempting custom load: {str(e)}")
                from tensorflow.keras.layers import DepthwiseConv2D
                def depthwise_conv2d_without_groups(**kwargs):
                    kwargs.pop('groups', None)
                    return DepthwiseConv2D(**kwargs)
                self.model = load_model(
                    MODEL_PATH,
                    compile=False,
                    custom_objects={'DepthwiseConv2D': depthwise_conv2d_without_groups}
                )

            # Fix input issues if needed
            if len(self.model.inputs) > 1:
                logger.info("🔧 Fixing multi-input model...")
                new_input = Input(shape=(224, 224, 3), name='fixed_input')
                output = self.model(new_input)
                self.model = Model(inputs=new_input, outputs=output)

            self.model.compile(optimizer=Adam(), 
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
            
            # Warmup
            self.model.predict(np.zeros((1, 224, 224, 3)))
            logger.info("✅ Model loaded and ready!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
            self.model = None

model_loader = ModelLoader()

@app.route('/health', methods=['GET'])
def health():
    model_loader.load_model()  # Attempt load if not tried yet
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
    app.run(host="0.0.0.0", port=5000)