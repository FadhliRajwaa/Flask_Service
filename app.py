import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Ini penting untuk integrasi dengan Node.js (cross-origin requests)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model-Retinopaty.h5')
model = None

def load_keras_model():
    global model
    try:
        logger.info(f"Mencoba memuat model dari: {MODEL_PATH}")
        
        if os.path.exists(MODEL_PATH):
            logger.info(f"File exist: {os.path.exists(MODEL_PATH)}")
            logger.info(f"File size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")
            
            model = load_model(MODEL_PATH)
            logger.info("Model berhasil dimuat!")
            return True
        else:
            logger.error(f"File model tidak ditemukan di {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error saat memuat model: {str(e)}")
        return False

# Fungsi untuk preprocessing gambar
def preprocess_image(img_data):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Resize image sesuai dengan input model (misalnya 224x224)
        img = img.resize((224, 224))
        
        # Convert ke array dan normalisasi
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalisasi
        
        return img_array
    except Exception as e:
        logger.error(f"Error saat preprocessing gambar: {str(e)}")
        return None

# Route untuk health check - untuk integrasi dengan Node.js
@app.route('/', methods=['GET'])
def index():
    model_status = "dimuat" if model is not None else "belum dimuat"
    file_exists = os.path.exists(MODEL_PATH)
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024) if file_exists else 0
    
    return jsonify({
        'status': 'success',
        'message': f'API Retinopati Diabetik berjalan dengan baik. Status model: {model_status}',
        'model_info': {
            'exists': file_exists,
            'path': MODEL_PATH,
            'size_mb': round(file_size, 2)
        }
    })

# Route untuk prediksi - endpoint utama yang akan diakses dari Node.js
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        # Coba muat model jika belum dimuat
        if not load_keras_model():
            return jsonify({
                'status': 'error',
                'message': 'Model belum dimuat dan gagal dimuat ulang'
            }), 500
    
    if 'image' not in request.json:
        return jsonify({
            'status': 'error',
            'message': 'Tidak ada gambar yang dikirim'
        }), 400
    
    try:
        img_data = request.json['image']
        processed_image = preprocess_image(img_data)
        
        if processed_image is None:
            return jsonify({
                'status': 'error',
                'message': 'Gagal memproses gambar'
            }), 400
        
        # Prediksi
        prediction = model.predict(processed_image)
        
        # Interpretasi hasil (sesuaikan dengan kelas model Anda)
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'class': class_names[predicted_class],
                'class_id': int(predicted_class),
                'confidence': confidence
            }
        })
    except Exception as e:
        logger.error(f"Error saat melakukan prediksi: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error saat melakukan prediksi: {str(e)}'
        }), 500

# Inisialisasi model saat aplikasi dijalankan
@app.before_first_request
def initialize():
    global model
    if model is None:
        load_keras_model()

if __name__ == '__main__':
    # Muat model saat aplikasi dimulai
    load_keras_model()
    
    # Jalankan aplikasi
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 