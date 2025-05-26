from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import datetime
import sys
import traceback
import io
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
import gc
import logging

# Load environment variables
load_dotenv()

# Kurangi log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Konfigurasi TensorFlow untuk membatasi penggunaan memori
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Nonaktifkan GPU untuk menghindari error CUDA
        tf.config.set_visible_devices([], 'GPU')
        logger.info("GPU dinonaktifkan untuk menghindari error CUDA")
    except Exception as e:
        logger.error(f"Error menonaktifkan GPU: {e}")
else:
    logger.info("Tidak ada GPU yang terdeteksi")

# Konfigurasi TensorFlow untuk membatasi penggunaan memori CPU
physical_devices = tf.config.list_physical_devices('CPU')
try:
    # Batasi penggunaan memori TensorFlow
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    # Jika gagal, coba cara alternatif
    logger.info("Tidak dapat mengatur memory growth, menggunakan metode alternatif")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Batasi ke 2GB memori per GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        )

app = Flask(__name__)
# Aktifkan CORS dengan konfigurasi yang lebih permisif
CORS(app, 
     resources={r"/*": {"origins": "*"}}, 
     supports_credentials=True, 
     allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     max_age=86400)

# Tambahkan header CORS tambahan untuk setiap respons
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With,Accept,Origin')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Konfigurasi model dan status
MODEL_LOADED = False
MODEL = None
ERROR_MESSAGE = None
STARTUP_TIME = datetime.datetime.now()
SIMULATION_MODE = os.environ.get('SIMULATION_MODE_ENABLED', 'false').lower() in ('true', '1', 'yes')

logger.info(f"Simulation mode: {'enabled' if SIMULATION_MODE else 'disabled'}")

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI')
try:
    if MONGO_URI:
        logger.info(f"Connecting to MongoDB...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Validasi koneksi
        db = client['retinopathy_prediction']
        predictions_collection = db['predictions']
        logger.info("MongoDB connected successfully!")
    else:
        logger.warning("MONGO_URI not set, using local storage for predictions")
        db = None
        predictions_collection = None
except Exception as e:
    logger.error(f"MongoDB connection failed: {str(e)}")
    logger.info("Using local storage for predictions")
    db = None
    predictions_collection = None

# Retinopathy class names
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Load Retinopathy model dengan penanganan error yang lebih baik
def load_ml_model():
    global MODEL, MODEL_LOADED, ERROR_MESSAGE
    
    # Jika mode simulasi diaktifkan, tidak perlu memuat model
    if SIMULATION_MODE:
        logger.info("Simulation mode is enabled, skipping model loading")
        MODEL_LOADED = True
        return
    
    # Coba beberapa path yang mungkin untuk model
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model-Retinopaty.h5'),
        './model-Retinopaty.h5',
        '/opt/render/project/src/model-Retinopaty.h5',
        os.path.join(os.getcwd(), 'model-Retinopaty.h5')
    ]
    
    # Jika environment variable MODEL_PATH diatur, tambahkan ke daftar
    if os.environ.get('MODEL_PATH'):
        possible_paths.insert(0, os.environ.get('MODEL_PATH'))
    
    for model_path in possible_paths:
        try:
            logger.info(f"Trying to load model from: {model_path}")
            if os.path.exists(model_path):
                logger.info(f"Model file found at {model_path}, loading...")
                
                # Gunakan opsi lebih efisien untuk memuat model
                try:
                    # Coba dengan compile=False untuk mempercepat loading
                    # Batasi thread untuk mengurangi penggunaan memori
                    with tf.device('/CPU:0'):
                        MODEL = load_model(model_path, compile=False)
                    logger.info("Model loaded with compile=False")
                except Exception as e1:
                    logger.error(f"Error loading with compile=False: {str(e1)}")
                    # Fallback ke loading normal
                    with tf.device('/CPU:0'):
                        MODEL = load_model(model_path)
                    logger.info("Model loaded with default options")
                
                MODEL_LOADED = True
                logger.info("Model loaded successfully!")
                return
            else:
                logger.warning(f"Model file not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            traceback.print_exc(file=sys.stdout)
    
    # Jika semua path gagal, catat error
    logger.error("All model loading attempts failed")
    ERROR_MESSAGE = "Model could not be loaded"
    MODEL_LOADED = False

# Panggil fungsi load model
try:
    # Coba load model dengan penanganan error yang lebih baik
    load_ml_model()
    # Segera jalankan garbage collection untuk membebaskan memori
    gc.collect()
except Exception as e:
    logger.error(f"Critical error in load_ml_model: {str(e)}")
    traceback.print_exc(file=sys.stdout)
    ERROR_MESSAGE = f"Critical error loading model: {str(e)}"

def prepare_image(img, target_size=(224, 224)):
    """
    Mempersiapkan gambar untuk prediksi
    """
    # Pastikan gambar dalam mode RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize gambar
    img = img.resize(target_size)
    
    # Konversi ke array
    img_array = image.img_to_array(img)
    
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalisasi
    img_array = img_array / 255.0
    
    return img_array

# Fungsi untuk simulasi prediksi
def simulate_prediction():
    """
    Menghasilkan prediksi simulasi untuk mode simulasi
    """
    import random
    
    # Weights to make some classes more common
    weights = [0.45, 0.2, 0.2, 0.1, 0.05]  # Probabilities for each class
    class_idx = random.choices(range(len(CLASS_NAMES)), weights=weights)[0]
    
    class_name = CLASS_NAMES[class_idx]
    confidence = 0.7 + (random.random() * 0.3)  # Between 0.7 and 1.0
    
    logger.info(f"Simulated prediction: {class_name} with confidence {confidence:.2f}")
    
    return class_name, confidence

@app.route('/predict', methods=['POST'])
def predict():
    start_time = datetime.datetime.now()
    
    # Check if simulation mode is enabled
    if SIMULATION_MODE:
        logger.info("Simulation mode is enabled, returning simulated prediction")
        
        # Simulate file processing delay (100-300ms)
        import time, random
        time.sleep(random.uniform(0.1, 0.3))
        
        # Generate simulated prediction
        class_name, confidence = simulate_prediction()
        
        processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
        
        # Save to MongoDB if available
        if predictions_collection:
            try:
                prediction_record = {
                    'class': class_name,
                    'confidence': confidence,
                    'filename': 'simulation.jpg',
                    'timestamp': datetime.datetime.now(),
                    'processing_time_ms': processing_time,
                    'simulation': True
                }
                result = predictions_collection.insert_one(prediction_record)
                prediction_id = str(result.inserted_id)
            except Exception as db_error:
                logger.error(f"Error saving simulation to MongoDB: {str(db_error)}")
                prediction_id = None
        else:
            prediction_id = None
            
        return jsonify({
            'id': prediction_id,
            'class': class_name,
            'confidence': confidence,
            'processing_time_ms': processing_time,
            'simulation_mode': True
        })
    
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded properly', 'simulation_mode': False}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'simulation_mode': False}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'simulation_mode': False}), 400
    
    try:
        prediction_id = None
        
        # Konversi FileStorage ke BytesIO
        file_bytes = file.read()
        img_io = io.BytesIO(file_bytes)
        
        # Reset pointer file untuk penggunaan selanjutnya jika diperlukan
        file.seek(0)
        
        # Gunakan BytesIO untuk load_img
        img = image.load_img(img_io, target_size=(224, 224))
        img_array = prepare_image(img)
        
        # Prediksi dengan model
        with tf.device('/CPU:0'):
            preds = MODEL.predict(img_array, verbose=0)
        class_idx = np.argmax(preds[0])
        class_name = CLASS_NAMES[class_idx]
        confidence = float(np.max(preds[0]))
        
        # Bersihkan memori
        del img_array
        del preds
        gc.collect()
        
        # Save prediction to MongoDB if available
        prediction_record = {
            'class': class_name,
            'confidence': confidence,
            'filename': file.filename,
            'timestamp': datetime.datetime.now(),
            'processing_time_ms': (datetime.datetime.now() - start_time).total_seconds() * 1000,
            'simulation': False
        }
        
        if predictions_collection:
            try:
                result = predictions_collection.insert_one(prediction_record)
                prediction_id = str(result.inserted_id)
            except Exception as db_error:
                logger.error(f"Error saving to MongoDB: {str(db_error)}")
        
        # Jalankan garbage collection untuk membebaskan memori
        gc.collect()
        
        return jsonify({
            'id': prediction_id,
            'class': class_name, 
            'confidence': confidence,
            'processing_time_ms': prediction_record['processing_time_ms'],
            'simulation_mode': False
        })
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        return jsonify({'error': str(e), 'simulation_mode': False}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    if not predictions_collection:
        return jsonify({'error': 'MongoDB not connected'}), 503
    
    try:
        limit = int(request.args.get('limit', 20))
        page = int(request.args.get('page', 1))
        skip = (page - 1) * limit
        
        # Get total count
        total = predictions_collection.count_documents({})
        
        # Get predictions with pagination
        cursor = predictions_collection.find({}).sort('timestamp', -1).skip(skip).limit(limit)
        predictions = []
        
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            doc['timestamp'] = doc['timestamp'].isoformat()
            predictions.append(doc)
            
        return jsonify({
            'predictions': predictions,
            'total': total,
            'page': page,
            'limit': limit,
            'totalPages': (total + limit - 1) // limit
        })
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    if not predictions_collection:
        return jsonify({'error': 'MongoDB not connected'}), 503
    
    try:
        # Get counts by retinopathy type
        pipeline = [
            {"$group": {"_id": "$class", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        results = list(predictions_collection.aggregate(pipeline))
        stats = {item['_id']: item['count'] for item in results}
        
        # Get total predictions
        total = predictions_collection.count_documents({})
        
        # Get recent predictions (last 24 hours)
        last_day = datetime.datetime.now() - datetime.timedelta(days=1)
        recent = predictions_collection.count_documents({"timestamp": {"$gt": last_day}})
        
        return jsonify({
            'by_class': stats,
            'total': total,
            'recent_24h': recent
        })
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """Endpoint to test if model is loaded correctly"""
    if SIMULATION_MODE:
        return jsonify({
            'status': 'success',
            'message': 'Running in simulation mode',
            'simulation_mode': True
        })
    elif MODEL_LOADED:
        # Kirim ringkasan singkat saja untuk menghindari response terlalu besar
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully',
            'model_type': str(type(MODEL).__name__),
            'simulation_mode': False
        })
    else:
        return jsonify({
            'status': 'error',
            'message': ERROR_MESSAGE or 'Model failed to load',
            'simulation_mode': False
        }), 500

@app.route('/health', methods=['GET', 'HEAD'])
def health_check_lightweight():
    """
    Endpoint health check ringan khusus untuk monitoring Render
    """
    return jsonify({
        'status': 'online',
        'service': 'retinopathy-api-health',
        'timestamp': datetime.datetime.now().isoformat(),
        'simulation_mode': SIMULATION_MODE
    }), 200

@app.route('/', methods=['GET'])
def health_check():
    """
    Endpoint health check utama dengan informasi detail tentang API
    """
    try:
        memory_info = {}
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = {
                'rss_mb': process.memory_info().rss / 1024 / 1024,
                'vms_mb': process.memory_info().vms / 1024 / 1024
            }
        except ImportError:
            memory_info = {'status': 'psutil not available'}
        
        uptime = datetime.datetime.now() - STARTUP_TIME
        
        return jsonify({
            'status': 'online',
            'service': 'retinopathy-api',
            'model_name': 'Retinopathy Detection Model',
            'model_loaded': MODEL_LOADED,
            'error_message': ERROR_MESSAGE,
            'classes': CLASS_NAMES,
            'api_version': '1.0.1',
            'tf_version': tf.__version__,
            'memory_usage': memory_info,
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m {uptime.seconds%60}s",
            'timestamp': datetime.datetime.now().isoformat(),
            'simulation_mode_enabled': SIMULATION_MODE
        })
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'service': 'retinopathy-api'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 