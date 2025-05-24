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
import psutil
from tensorflow.keras import layers

# Konfigurasi logging dengan format yang lebih jelas untuk Render
logging.basicConfig(
    level=logging.INFO,
    format='[RETINOPATY-API] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup yang jelas
logger.info("======= STARTING RETINOPATY-API SERVICE =======")
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Python version: {sys.version}")
logger.info(f"NumPy version: {np.__version__}")

# Log informasi sistem
mem = psutil.virtual_memory()
logger.info(f"Total memory: {mem.total / (1024 * 1024 * 1024):.2f} GB")
logger.info(f"Available memory: {mem.available / (1024 * 1024 * 1024):.2f} GB")
logger.info(f"Memory usage: {mem.percent}%")

# Custom layer untuk kompatibilitas dengan model yang dibuat di versi TensorFlow yang berbeda
class CustomInputLayer(tf.keras.layers.Layer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            input_shape = batch_shape[1:]
            kwargs['input_shape'] = input_shape
        super(CustomInputLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model-Retinopaty.h5')
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Model exists: {os.path.exists(MODEL_PATH)}")
if os.path.exists(MODEL_PATH):
    logger.info(f"Model size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")

# Class names untuk model Retinopati
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Model akan dimuat di global scope
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
        
        # Setup custom objects untuk kompatibilitas
        custom_objects = {
            'CustomInputLayer': CustomInputLayer,
            'InputLayer': CustomInputLayer
        }
        
        # Log status memori sebelum memuat model
        memory_before = psutil.virtual_memory()
        logger.info(f"Memory before loading: {memory_before.percent}% used")
        
        logger.info("Mulai memuat model...")
        
        # Coba dengan pendekatan multiple approach
        try:
            # Pendekatan 1: Menggunakan custom objects
            logger.info("Pendekatan 1: Menggunakan custom objects...")
            model = load_model(
                MODEL_PATH,
                custom_objects=custom_objects,
                compile=False
            )
        except Exception as e1:
            logger.warning(f"Pendekatan 1 gagal: {str(e1)}")
            
            # Jika gagal, coba pendekatan lain
            try:
                logger.info("Pendekatan 2: Menggunakan hanya model weights...")
                
                # Coba buat model dari scratch berdasarkan dimensi
                input_shape = (224, 224, 3)  # Ukuran umum untuk model retinopati
                
                # Memuat model sebagai full model dengan safe_mode=False untuk TF 2.14.0
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                
            except Exception as e2:
                logger.warning(f"Pendekatan 2 gagal: {str(e2)}")
                
                try:
                    logger.info("Pendekatan 3: Menggunakan TensorFlow SavedModel...")
                    # Coba memuat sebagai SavedModel
                    # Jika model dalam format H5 tetapi menggunakan fitur newer TensorFlow 
                    # Simpan ulang model sementara sebagai SavedModel
                    temp_saved_model_dir = os.path.join(os.path.dirname(MODEL_PATH), 'temp_model')
                    if not os.path.exists(temp_saved_model_dir):
                        os.makedirs(temp_saved_model_dir)
                    
                    # Jika gagal, coba konversi menggunakan tf.saved_model.load
                    model = tf.saved_model.load(MODEL_PATH)
                    
                except Exception as e3:
                    logger.error(f"Semua pendekatan gagal. Pendekatan 3 error: {str(e3)}")
                    raise Exception(f"Tidak dapat memuat model dengan semua pendekatan. Versi TensorFlow mungkin terlalu berbeda. Model dibuat dengan TF versi lain, server menggunakan {tf.__version__}")
        
        # Log status memori setelah memuat model
        memory_after = psutil.virtual_memory()
        logger.info(f"Memory after loading: {memory_after.percent}% used")
        logger.info(f"Memory increase: {memory_after.percent - memory_before.percent}%")
        
        if model is not None:
            if hasattr(model, 'input_shape'):
                logger.info(f"Model berhasil dimuat! Model shape: {model.input_shape}")
            else:
                logger.info("Model berhasil dimuat! (Tidak dapat menentukan shape)")
            
            logger.info("=== LOADING MODEL SELESAI ===")
            return True
        else:
            logger.error("Model tidak berhasil dimuat, nilai model adalah None")
            return False
            
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

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)  # Ini penting untuk integrasi dengan Node.js (cross-origin requests)

# Route untuk health check - untuk integrasi dengan Node.js
@app.route('/', methods=['GET'])
def index():
    file_exists = os.path.exists(MODEL_PATH)
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024) if file_exists else 0
    
    # Status model yang sebenarnya
    model_status = "not_loaded" if model is None else "loaded"
    
    # Mendapatkan informasi model
    model_info = {
        'exists': file_exists,
        'path': MODEL_PATH,
        'size_mb': round(file_size, 2),
        'status': model_status,
        'tf_version': tf.__version__
    }
    
    # Tambahkan informasi input/output shape jika model sudah dimuat
    if model is not None and hasattr(model, 'input_shape'):
        model_info['input_shape'] = str(model.input_shape)
        if hasattr(model, 'output_shape'):
            model_info['output_shape'] = str(model.output_shape)
    
    return jsonify({
        'status': 'success',
        'message': 'API Retinopati Diabetik berjalan dengan baik',
        'model_info': model_info,
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
        model_info = {'success': True, 'message': 'Model berhasil dimuat'}
        
        # Tambahkan detail model jika tersedia
        if hasattr(model, 'input_shape'):
            model_info['input_shape'] = str(model.input_shape)
        if hasattr(model, 'output_shape'):
            model_info['output_shape'] = str(model.output_shape)
            
        return jsonify({
            'status': 'success',
            'message': 'Model berhasil dimuat',
            'model_info': model_info
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
        
        # 2. Memastikan model telah dimuat
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
            # Prediksi dengan handling untuk model yang berbeda
            logger.info("Mulai proses prediksi")
            
            # Cek apakah model memiliki metode predict
            if hasattr(model, 'predict'):
                # Handle model dengan input_shape yang berbeda jika perlu
                if hasattr(model, 'input_shape'):
                    input_shape = model.input_shape[1:]
                    logger.info(f"Model input shape: {input_shape}")
                    logger.info(f"Processed image shape: {processed_image.shape}")
                    
                    # Pastikan gambar sesuai dengan input model
                    if processed_image.shape[1:] != input_shape:
                        processed_image = tf.image.resize(processed_image, (input_shape[0], input_shape[1]))
                        logger.info(f"Resized to match model input: {processed_image.shape}")
                
                # Prediksi menggunakan metode predict
                prediction = model.predict(processed_image)
            else:
                # Untuk model yang tidak memiliki metode predict
                logger.info("Model tidak memiliki metode predict, mencoba alternatif")
                # Gunakan panggilan model langsung sebagai fungsi
                prediction = model(processed_image)
                
                # Konversi ke numpy array jika berupa tensor
                if hasattr(prediction, 'numpy'):
                    prediction = prediction.numpy()
            
            logger.info("Prediksi selesai")
            
            # Interpretasi hasil
            if isinstance(prediction, np.ndarray) and prediction.size > 0:
                if len(prediction.shape) > 1:
                    predicted_class = np.argmax(prediction[0])
                    confidence = float(prediction[0][predicted_class])
                else:
                    predicted_class = np.argmax(prediction)
                    confidence = float(prediction[predicted_class])
                
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
            else:
                logger.error("Format output prediksi tidak sesuai ekspektasi")
                return jsonify({
                    'status': 'error',
                    'message': 'Format output prediksi tidak sesuai ekspektasi'
                }), 500
                
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
        # Tidak perlu membersihkan model, kita ingin mempertahankannya di memori
        pass

# Memuat model saat aplikasi dimulai
logger.info("==================================================")
logger.info("INISIALISASI APLIKASI: Mencoba memuat model saat startup...")
logger.info("==================================================")
try:
    # Log status memori sebelum loading
    mem_before = psutil.virtual_memory()
    logger.info(f"[STARTUP] Memory sebelum loading model: {mem_before.percent}% used ({mem_before.available / (1024**3):.2f} GB available)")
    
    # Memuat model saat startup
    success = load_model_from_file()
    
    # Log status memori setelah loading
    mem_after = psutil.virtual_memory()
    logger.info(f"[STARTUP] Memory setelah loading model: {mem_after.percent}% used ({mem_after.available / (1024**3):.2f} GB available)")
    logger.info(f"[STARTUP] Perubahan penggunaan memori: {mem_after.percent - mem_before.percent}%")
    
    if success:
        logger.info("==================================================")
        logger.info("MODEL BERHASIL DIMUAT PADA STARTUP!")
        logger.info("==================================================")
    else:
        logger.info("==================================================")
        logger.info("MODEL GAGAL DIMUAT PADA STARTUP! AKAN DICOBA LAGI SAAT REQUEST PERTAMA")
        logger.info("==================================================")
except Exception as e:
    logger.error(f"Error saat memuat model pada startup: {str(e)}")
    logger.error("==================================================")
    logger.error("MODEL GAGAL DIMUAT PADA STARTUP! AKAN DICOBA LAGI SAAT REQUEST PERTAMA")
    logger.error("==================================================")

if __name__ == '__main__':
    logger.info("Starting Flask app in development mode")
    # Port dan host untuk development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 