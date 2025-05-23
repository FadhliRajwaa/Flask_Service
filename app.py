from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import os
import platform
import sys
import json

# Coba import TensorFlow, tetapi jangan gagal jika tidak ada
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("TensorFlow imported successfully")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, running in simulation mode only")

app = Flask(__name__)
CORS(app, origins=['*'], supports_credentials=True, methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

# Cek apakah mode simulasi diaktifkan
simulation_mode = os.environ.get("SIMULATION_MODE") == "1" or not TF_AVAILABLE
print(f"Simulation mode: {'ON' if simulation_mode else 'OFF'}")
print(f"Reason: {'SIMULATION_MODE=1' if os.environ.get('SIMULATION_MODE') == '1' else 'TensorFlow not available' if not TF_AVAILABLE else 'Unknown'}")

# Load trained model hanya jika tidak dalam mode simulasi dan TensorFlow tersedia
model = None
if not simulation_mode and TF_AVAILABLE:
    try:
        print("Attempting to load model...")
        model = load_model("model-Retinopaty.h5")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to simulation mode")
        simulation_mode = True
else:
    print("Running in simulation mode, model not loaded")

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
    # Gunakan variabel global untuk mode simulasi
    global simulation_mode
    
    # Log request untuk debugging
    print(f"Received prediction request. Simulation mode: {simulation_mode}")
    print(f"Request files: {list(request.files.keys()) if request.files else 'No files'}")
    
    # Terima file dengan nama 'file' atau 'image'
    file_key = None
    if "file" in request.files:
        file_key = "file"
    elif "image" in request.files:
        file_key = "image"
    
    if file_key is None and not simulation_mode:
        print("No image file provided and not in simulation mode")
        # Aktifkan mode simulasi sebagai fallback
        simulation_mode = True
        print("Activating simulation mode as fallback")

    try:
        if not simulation_mode and model is not None and file_key is not None:
            # Mode normal dengan model
            print("Using actual model for prediction")
            image_file = request.files[file_key]
            image = Image.open(io.BytesIO(image_file.read()))
            input_tensor = preprocess_image(image)
            
            predictions = model.predict(input_tensor)[0]
            class_index = predictions.argmax()
            class_name = CLASSES[class_index]
            confidence = float(predictions[class_index])
            print(f"Model prediction: {class_name} with confidence {confidence}")
        else:
            # Mode simulasi
            print("Using simulation mode for prediction")
            import random
            class_index = random.randint(0, len(CLASSES)-1)
            class_name = CLASSES[class_index]
            confidence = 0.7 + random.random() * 0.3  # 0.7-1.0
            print(f"Simulation prediction: {class_name} with confidence {confidence}")

        # Tambahkan informasi tambahan untuk debugging
        simulation_info = {
            "is_simulation": simulation_mode or model is None,
            "reason": "SIMULATION_MODE=1 in environment" if os.environ.get("SIMULATION_MODE") == "1" else 
                     "Model not loaded" if model is None else 
                     "No file provided" if file_key is None else None
        }
        
        response_data = {
            "severity": class_name,
            "severity_level": SEVERITY_LEVEL_MAPPING[class_name],
            "confidence": confidence,
            "severity_description": SEVERITY_MAPPING[class_name],
            "recommendation": RECOMMENDATIONS[class_name],
            "raw_prediction": simulation_info
        }
        
        print(f"Sending response: {response_data}")
        return jsonify(response_data)
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
    global simulation_mode
    
    response_data = {
        "status": "ok",
        "model_name": "model-Retinopaty.h5",
        "model_loaded": model_loaded,
        "classes": CLASSES,
        "severity_mapping": SEVERITY_MAPPING,
        "severity_level_mapping": SEVERITY_LEVEL_MAPPING,
        "simulation_mode": simulation_mode,
        "simulation_reason": "SIMULATION_MODE=1 in environment" if os.environ.get("SIMULATION_MODE") == "1" else
                           "TensorFlow not available" if not TF_AVAILABLE else
                           "Model not loaded" if not model_loaded else None,
        "api_version": "1.0.0",
        "platform": platform.platform(),
        "python_version": sys.version
    }
    
    # Tambahkan tf_version hanya jika TensorFlow tersedia
    if TF_AVAILABLE:
        response_data["tf_version"] = tf.__version__
    
    print(f"Info endpoint called, returning: {json.dumps(response_data, indent=2)}")
    return jsonify(response_data)

@app.route("/", methods=["GET"])
def home():
    """Root endpoint untuk health check."""
    model_loaded = model is not None
    global simulation_mode
    
    response_data = {
        "status": "ok",
        "message": "Flask API for RetinaScan is running",
        "model_loaded": model_loaded,
        "simulation_mode": simulation_mode,
        "tensorflow_available": TF_AVAILABLE,
        "environment": {
            "SIMULATION_MODE": os.environ.get("SIMULATION_MODE"),
            "TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
            "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL"),
            "PORT": os.environ.get("PORT")
        }
    }
    
    print(f"Health check endpoint called, returning: {json.dumps(response_data, indent=2)}")
    return jsonify(response_data)

@app.after_request
def after_request(response):
    # Log response untuk debugging
    print(f"Sending response with status {response.status_code}")
    
    # Tambahkan header CORS
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Handle OPTIONS request untuk preflight CORS
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    print(f"Handling OPTIONS request for path: /{path}")
    response = app.make_default_options_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Tambahkan endpoint khusus untuk testing
@app.route("/test", methods=["GET"])
def test():
    """Endpoint untuk testing koneksi."""
    import time
    print("Test endpoint called")
    return jsonify({
        "status": "ok",
        "message": "Flask API test endpoint is working",
        "timestamp": time.time()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask API on port {port}")
    print(f"Simulation mode: {'ON' if simulation_mode else 'OFF'}")
    print(f"TensorFlow available: {'YES' if TF_AVAILABLE else 'NO'}")
    print(f"Model loaded: {'YES' if model is not None else 'NO'}")
    app.run(debug=True, host='0.0.0.0', port=port)
