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

# Konfigurasi TensorFlow untuk membatasi penggunaan memori
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Nonaktifkan GPU untuk menghindari error CUDA
        tf.config.set_visible_devices([], 'GPU')
        print("GPU dinonaktifkan untuk menghindari error CUDA")
    except Exception as e:
        print(f"Error menonaktifkan GPU: {e}")
else:
    print("Tidak ada GPU yang terdeteksi")

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

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI')
try:
    if MONGO_URI:
        print(f"Connecting to MongoDB...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Validasi koneksi
        db = client['retinopathy_prediction']
        predictions_collection = db['predictions']
        print("MongoDB connected successfully!")
    else:
        print("MONGO_URI not set, using local storage for predictions")
        db = None
        predictions_collection = None
except Exception as e:
    print(f"MongoDB connection failed: {str(e)}")
    print("Using local storage for predictions")
    db = None
    predictions_collection = None

# Retinopathy class names
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Load Retinopathy model dengan penanganan error yang lebih baik
def load_ml_model():
    global MODEL, MODEL_LOADED, ERROR_MESSAGE
    
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
            print(f"Trying to load model from: {model_path}")
            if os.path.exists(model_path):
                print(f"Model file found at {model_path}, loading...")
                
                # Gunakan opsi lebih efisien untuk memuat model
                try:
                    # Coba dengan compile=False untuk mempercepat loading
                    MODEL = load_model(model_path, compile=False)
                    print("Model loaded with compile=False")
                except Exception as e1:
                    print(f"Error loading with compile=False: {str(e1)}")
                    # Fallback ke loading normal
                    MODEL = load_model(model_path)
                    print("Model loaded with default options")
                
                MODEL_LOADED = True
                print("Model loaded successfully!")
                return
            else:
                print(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            traceback.print_exc(file=sys.stdout)
    
    # Jika semua path gagal, catat error
    print("All model loading attempts failed")
    ERROR_MESSAGE = "Model could not be loaded"
    MODEL_LOADED = False

# Panggil fungsi load model
try:
    load_ml_model()
except Exception as e:
    print(f"Critical error in load_ml_model: {str(e)}")
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

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded properly'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
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
            'timestamp': datetime.datetime.now()
        }
        
        if predictions_collection:
            try:
                result = predictions_collection.insert_one(prediction_record)
                prediction_id = str(result.inserted_id)
            except Exception as db_error:
                print(f"Error saving to MongoDB: {str(db_error)}")
        
        # Jalankan garbage collection untuk membebaskan memori
        gc.collect()
        
        return jsonify({
            'id': prediction_id,
            'class': class_name, 
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        return jsonify({'error': str(e)}), 500

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
        return jsonify({'error': str(e)}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """Endpoint to test if model is loaded correctly"""
    if MODEL_LOADED:
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully',
            'model_summary': str(MODEL.summary())
        })
    else:
        return jsonify({
            'status': 'error',
            'message': ERROR_MESSAGE or 'Model failed to load'
        }), 500

@app.route('/health', methods=['GET', 'HEAD'])
def health_check_lightweight():
    """
    Endpoint health check ringan khusus untuk monitoring Render
    """
    return jsonify({
        'status': 'online',
        'service': 'retinopathy-api-health',
        'timestamp': datetime.datetime.now().isoformat()
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
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'service': 'retinopathy-api'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 