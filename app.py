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
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Konfigurasi model dan status
MODEL_LOADED = False
SIMULATION_MODE = False
MODEL = None
ERROR_MESSAGE = None

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://rajwaarahmana45:123abc789@cluster0.cp7fh.mongodb.net/RetinaScan?retryWrites=true&w=majority')
try:
    print(f"Connecting to MongoDB: {MONGO_URI[:20]}...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # Validasi koneksi
    db = client['retinopathy_prediction']
    predictions_collection = db['predictions']
    print("MongoDB connected successfully!")
except Exception as e:
    print(f"MongoDB connection failed: {str(e)}")
    print("Using local storage for predictions")
    db = None
    predictions_collection = None

# Retinopathy class names
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Load Retinopathy model dengan penanganan error yang lebih baik
def load_ml_model():
    global MODEL, MODEL_LOADED, SIMULATION_MODE, ERROR_MESSAGE
    
    # Coba beberapa path yang mungkin untuk model
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model-Retinopaty.h5'),
        './model-Retinopaty.h5',
        '/opt/render/project/src/model-Retinopaty.h5',
        os.path.join(os.getcwd(), 'model-Retinopaty.h5')
    ]
    
    for model_path in possible_paths:
        try:
            print(f"Trying to load model from: {model_path}")
            if os.path.exists(model_path):
                print(f"Model file found at {model_path}, loading...")
                MODEL = load_model(model_path)
                MODEL_LOADED = True
                print("Model loaded successfully!")
                return
            else:
                print(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            traceback.print_exc(file=sys.stdout)
    
    # Jika semua path gagal, aktifkan mode simulasi
    print("All model loading attempts failed, activating SIMULATION MODE")
    SIMULATION_MODE = True
    ERROR_MESSAGE = "Model could not be loaded, using simulation mode"

# Panggil fungsi load model
load_ml_model()

def prepare_image(img, target_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def simulate_prediction():
    """Fungsi untuk mensimulasikan prediksi ketika model tidak tersedia"""
    # Pilih kelas secara acak dengan distribusi yang masuk akal
    class_probabilities = [0.5, 0.2, 0.15, 0.1, 0.05]  # Lebih banyak kasus normal
    class_idx = np.random.choice(len(CLASS_NAMES), p=class_probabilities)
    class_name = CLASS_NAMES[class_idx]
    
    # Buat confidence yang masuk akal berdasarkan kelas
    if class_idx == 0:  # No DR
        confidence = np.random.uniform(0.85, 0.99)
    elif class_idx == 1:  # Mild
        confidence = np.random.uniform(0.75, 0.9)
    elif class_idx == 2:  # Moderate
        confidence = np.random.uniform(0.7, 0.85)
    elif class_idx == 3:  # Severe
        confidence = np.random.uniform(0.75, 0.9)
    else:  # Proliferative DR
        confidence = np.random.uniform(0.8, 0.95)
    
    return class_name, float(confidence)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED and not SIMULATION_MODE:
        return jsonify({'error': 'Model not loaded properly'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        prediction_id = None
        
        # Jika model dimuat, gunakan model untuk prediksi
        if MODEL_LOADED:
            img = image.load_img(file, target_size=(224, 224))
            img_array = prepare_image(img)
            preds = MODEL.predict(img_array)
            class_idx = np.argmax(preds[0])
            class_name = CLASS_NAMES[class_idx]
            confidence = float(np.max(preds[0]))
        # Jika tidak, gunakan mode simulasi
        else:
            class_name, confidence = simulate_prediction()
        
        # Save prediction to MongoDB if available
        prediction_record = {
            'class': class_name,
            'confidence': confidence,
            'filename': file.filename,
            'timestamp': datetime.datetime.now(),
            'simulation_mode': SIMULATION_MODE
        }
        
        if predictions_collection:
            try:
                result = predictions_collection.insert_one(prediction_record)
                prediction_id = str(result.inserted_id)
            except Exception as db_error:
                print(f"Error saving to MongoDB: {str(db_error)}")
        
        return jsonify({
            'id': prediction_id,
            'class': class_name, 
            'confidence': confidence,
            'simulation_mode': SIMULATION_MODE,
            'raw_prediction': {
                'is_simulation': SIMULATION_MODE
            }
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        
        # Fallback ke simulasi jika terjadi error
        try:
            class_name, confidence = simulate_prediction()
            return jsonify({
                'id': None,
                'class': class_name, 
                'confidence': confidence,
                'simulation_mode': True,
                'error_info': str(e),
                'raw_prediction': {
                    'is_simulation': True,
                    'fallback_due_to_error': True
                }
            })
        except:
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
            'message': ERROR_MESSAGE or 'Model failed to load',
            'simulation_mode': SIMULATION_MODE
        }), 200  # Return 200 even in simulation mode

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    # Cek status sistem
    system_info = {
        'python_version': sys.version,
        'tensorflow_version': tf.__version__,
        'working_directory': os.getcwd(),
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('TF_', 'PYTHON', 'PATH'))},
    }
    
    # Cek status model
    model_info = {}
    if MODEL_LOADED:
        try:
            model_info = {
                'model_type': str(type(MODEL)),
                'input_shape': str(MODEL.input_shape),
                'output_shape': str(MODEL.output_shape),
                'layers_count': len(MODEL.layers),
            }
        except:
            model_info = {'model_info_error': 'Could not retrieve model details'}
    
    return jsonify({
        'status': 'online',
        'service': 'retinopathy-api',
        'model_loaded': MODEL_LOADED,
        'model_name': 'Retinopathy Detection Model',
        'model_info': model_info if MODEL_LOADED else None,
        'classes': CLASS_NAMES,
        'tf_version': tf.__version__,
        'simulation_mode': SIMULATION_MODE,
        'api_version': '1.0.1',
        'error_message': ERROR_MESSAGE,
        'system_info': system_info
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 