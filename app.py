import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import logging
import sys
import gc
import tensorflow as tf

# Konfigurasi logging dengan format yang lebih jelas untuk Render
logging.basicConfig(
    level=logging.INFO,
    format='[RETINOPATY-API] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup yang jelas
logger.info("======= STARTING RETINOPATY-API SERVICE =======")

app = Flask(__name__)
CORS(app)  # Ini penting untuk integrasi dengan Node.js (cross-origin requests)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model-Retinopaty.h5')
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Model exists: {os.path.exists(MODEL_PATH)}")
if os.path.exists(MODEL_PATH):
    logger.info(f"Model size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")

# Class names untuk model Retinopati
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Model akan dimuat hanya saat diperlukan
model = None

def load_model_from_file():
    global model
    try:
        logger.info("=== MEMULAI PROSES LOADING MODEL ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"NumPy version: {np.__version__}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model tidak ditemukan di {MODEL_PATH}")
            return False
                
        logger.info(f"Model file size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")
        
        # Import yang diperlukan
        from tensorflow.keras.models import load_model
        
        # Coba dengan custom_objects kosong
        custom_objects = {}
        
        # Log status memori sebelum memuat model
        import psutil
        memory_before = psutil.virtual_memory()
        logger.info(f"Memory before loading: {memory_before.percent}% used")
        
        logger.info("Mulai memuat model...")
        # Load model dengan explicit settings
        model = load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Log status memori setelah memuat model
        memory_after = psutil.virtual_memory()
        logger.info(f"Memory after loading: {memory_after.percent}% used")
        logger.info(f"Memory increase: {memory_after.percent - memory_before.percent}%")
        
        logger.info(f"Model berhasil dimuat! Model shape: {model.input_shape}")
        logger.info("=== LOADING MODEL SELESAI ===")
        return True
    except Exception as e:
        logger.error(f"Error saat memuat model: {str(e)}")
        logger.error("=== LOADING MODEL GAGAL ===")
        return False

# Fungsi untuk preprocessing gambar
def preprocess_image(img_data):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Resize image sesuai dengan input model (misalnya 224x224)
        img = img.resize((224, 224))
        logger.info(f"Image resized to: {img.size}")
        
        # Convert ke array dan normalisasi
        img_array = np.array(img, dtype=np.float32)
        
        # Handle untuk gambar RGBA (with alpha channel)
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]  # Keep only RGB
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalisasi
        
        return img_array
    except Exception as e:
        logger.error(f"Error saat preprocessing gambar: {str(e)}")
        return None

# Route untuk health check - untuk integrasi dengan Node.js
@app.route('/', methods=['GET'])
def index():
    file_exists = os.path.exists(MODEL_PATH)
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024) if file_exists else 0
    
    # Membuat model dan menyimpan status, namun tanpa memuat model penuh
    model_status = "not_loaded" if model is None else "loaded"
    
    return jsonify({
        'status': 'success',
        'message': 'API Retinopati Diabetik berjalan dengan baik',
        'model_info': {
            'exists': file_exists,
            'path': MODEL_PATH,
            'size_mb': round(file_size, 2),
            'status': model_status
        },
        'system_info': {
            'python_version': sys.version,
            'tensorflow_version': tf.__version__,
        }
    })

# Route untuk menguji pemuatan model
@app.route('/load-model', methods=['GET'])
def test_load_model():
    global model
    
    # Jika model sudah dimuat, bersihkan terlebih dahulu
    if model is not None:
        model = None
        gc.collect()
        tf.keras.backend.clear_session()
        logger.info("Model existing dibersihkan dari memori")
    
    # Coba muat model
    success = load_model_from_file()
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Model berhasil dimuat',
            'model_info': {
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape)
            }
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Gagal memuat model'
        }), 500

# Route untuk prediksi - endpoint utama yang akan diakses dari Node.js
@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Menerima request prediksi")
    
    if 'image' not in request.json:
        logger.error("Tidak ada gambar dalam request")
        return jsonify({
            'status': 'error',
            'message': 'Tidak ada gambar yang dikirim'
        }), 400
    
    try:
        # 1. Preprocessing gambar
        logger.info("Memulai preprocessing gambar")
        img_data = request.json['image']
        processed_image = preprocess_image(img_data)
        
        if processed_image is None:
            logger.error("Gagal memproses gambar")
            return jsonify({
                'status': 'error',
                'message': 'Gagal memproses gambar'
            }), 400
        
        # 2. Memuat model jika belum dimuat
        global model
        if model is None:
            logger.info("Model belum dimuat, mulai proses loading")
            success = load_model_from_file()
            if not success:
                logger.error("Gagal memuat model")
                return jsonify({
                    'status': 'error',
                    'message': 'Gagal memuat model'
                }), 500
        else:
            logger.info("Model sudah dimuat sebelumnya")
        
        # 3. Melakukan prediksi
        try:
            # Memastikan tensor memiliki shape yang benar
            input_shape = model.input_shape[1:]
            
            # Log untuk debugging
            logger.info(f"Model input shape: {input_shape}")
            logger.info(f"Processed image shape: {processed_image.shape}")
            
            # Pastikan gambar sesuai dengan input model
            if processed_image.shape[1:] != input_shape:
                processed_image = tf.image.resize(processed_image, (input_shape[0], input_shape[1]))
                logger.info(f"Resized to match model input: {processed_image.shape}")
            
            # Prediksi
            logger.info("Mulai proses prediksi")
            prediction = model.predict(processed_image)
            logger.info("Prediksi selesai")
            
            # Interpretasi hasil
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            logger.info(f"Hasil prediksi: Class={CLASS_NAMES[predicted_class]}, Confidence={confidence:.4f}")
            
            # Respon dengan hasil prediksi
            return jsonify({
                'status': 'success',
                'prediction': {
                    'class': CLASS_NAMES[predicted_class],
                    'class_id': int(predicted_class),
                    'confidence': confidence
                }
            })
        except Exception as pred_error:
            logger.error(f"Error during prediction: {str(pred_error)}")
            return jsonify({
                'status': 'error',
                'message': f'Error saat melakukan prediksi: {str(pred_error)}'
            }), 500
    except Exception as e:
        logger.error(f"Error saat memproses request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error saat memproses request: {str(e)}'
        }), 500
    finally:
        # Clean up untuk mengurangi memory usage
        logger.info("Membersihkan sesi TensorFlow")
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == '__main__':
    logger.info("Starting Flask app in development mode")
    # Tidak memuat model saat startup untuk menghemat memori
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 