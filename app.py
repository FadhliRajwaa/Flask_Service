from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import platform
import sys

app = Flask(__name__)
CORS(app, origins=['*'], supports_credentials=True, methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

# Load trained model
try:
    model = load_model("model-Retinopaty.h5")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback untuk mode simulasi
    model = None

# Kelas output model (5 kelas untuk tingkat keparahan DR)
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Mapping output ke bahasa Indonesia
SEVERITY_MAPPING = {
    'No DR': 'Tidak ada',
    'Mild': 'Ringan',
    'Moderate': 'Sedang',
    'Severe': 'Berat',
    'Proliferative DR': 'Sangat Berat'
}

# Mapping tingkat keparahan numerik
SEVERITY_LEVEL_MAPPING = {
    'No DR': 0,
    'Mild': 1,
    'Moderate': 2,
    'Severe': 3,
    'Proliferative DR': 4
}

# Rekomendasi berdasarkan tingkat keparahan
RECOMMENDATIONS = {
    'No DR': 'Lakukan pemeriksaan rutin setiap tahun.',
    'Mild': 'Kontrol gula darah dan tekanan darah. Pemeriksaan ulang dalam 9-12 bulan.',
    'Moderate': 'Konsultasi dengan dokter spesialis mata. Pemeriksaan ulang dalam 6 bulan.',
    'Severe': 'Rujukan segera ke dokter spesialis mata. Pemeriksaan ulang dalam 2-3 bulan.',
    'Proliferative DR': 'Rujukan segera ke dokter spesialis mata untuk evaluasi dan kemungkinan tindakan laser atau operasi.'
}

# Preprocessing image
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalisasi
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/predict", methods=["POST"])
def predict():
    # Cek mode simulasi
    simulation_mode = os.environ.get("SIMULATION_MODE") == "1"
    
    if "file" not in request.files and not simulation_mode:
        return jsonify({"error": "No image file provided"}), 400

    try:
        if not simulation_mode and model is not None:
            # Mode normal dengan model
            image_file = request.files["file"]
            image = Image.open(io.BytesIO(image_file.read()))
            input_tensor = preprocess_image(image)
            
            predictions = model.predict(input_tensor)[0]
            class_index = predictions.argmax()
            class_name = CLASSES[class_index]
            confidence = float(predictions[class_index])
        else:
            # Mode simulasi
            import random
            class_index = random.randint(0, len(CLASSES)-1)
            class_name = CLASSES[class_index]
            confidence = 0.7 + random.random() * 0.3  # 0.7-1.0

        # Tambahkan informasi tambahan untuk debugging
        simulation_info = {
            "is_simulation": simulation_mode or model is None,
            "reason": "SIMULATION_MODE=1 in environment" if os.environ.get("SIMULATION_MODE") == "1" else 
                     "Model not loaded" if model is None else None
        }
        
        return jsonify({
            "severity": class_name,
            "severity_level": SEVERITY_LEVEL_MAPPING[class_name],
            "confidence": confidence,
            "severity_description": SEVERITY_MAPPING[class_name],
            "recommendation": RECOMMENDATIONS[class_name],
            "raw_prediction": simulation_info
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "severity": "Moderate",  # Default fallback
            "severity_level": 2,
            "confidence": 0.8,
            "severity_description": SEVERITY_MAPPING["Moderate"],
            "recommendation": RECOMMENDATIONS["Moderate"],
            "raw_prediction": {
                "is_simulation": True,
                "reason": f"Error during prediction: {str(e)}"
            }
        })

@app.route("/info", methods=["GET"])
def info():
    """Endpoint untuk mendapatkan informasi tentang model dan API."""
    model_loaded = model is not None
    simulation_mode = os.environ.get("SIMULATION_MODE") == "1" or not model_loaded
    
    return jsonify({
        "status": "ok",
        "model_name": "model-Retinopaty.h5",
        "model_loaded": model_loaded,
        "classes": CLASSES,
        "severity_mapping": SEVERITY_MAPPING,
        "severity_level_mapping": SEVERITY_LEVEL_MAPPING,
        "simulation_mode": simulation_mode,
        "simulation_reason": "SIMULATION_MODE=1 in environment" if os.environ.get("SIMULATION_MODE") == "1" else
                           "Model not loaded" if not model_loaded else None,
        "api_version": "1.0.0",
        "tf_version": tf.__version__,
        "platform": platform.platform(),
        "python_version": sys.version
    })

@app.route("/", methods=["GET"])
def home():
    """Root endpoint untuk health check."""
    model_loaded = model is not None
    simulation_mode = os.environ.get("SIMULATION_MODE") == "1" or not model_loaded
    
    return jsonify({
        "status": "ok",
        "message": "Flask API for RetinaScan is running",
        "model_loaded": model_loaded,
        "simulation_mode": simulation_mode,
        "environment": {
            "SIMULATION_MODE": os.environ.get("SIMULATION_MODE"),
            "TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
            "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL"),
            "PORT": os.environ.get("PORT")
        }
    })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask API on port {port}")
    print(f"Simulation mode: {'ON' if os.environ.get('SIMULATION_MODE') == '1' else 'OFF'}")
    print(f"Model loaded: {'YES' if model is not None else 'NO'}")
    app.run(debug=True, host='0.0.0.0', port=port)
