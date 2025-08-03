from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import base64
import cv2
import numpy as np
import io
from PIL import Image
import threading
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('model', 'model.h5')
LABELS_PATH = os.path.join('model', 'labels.txt')

model = None
class_names = []

def load_model_lazy():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load labels once
try:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded {len(class_names)} classes")
except Exception as e:
    print(f"❌ Label loading failed: {str(e)}")

def preprocess_image(image_file):
    img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
@app.route('/health')
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    load_model_lazy()
    
    if not model or not class_names:
        return jsonify({"error": "Model not ready"}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
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

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    from os import getenv
    port = int(getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
