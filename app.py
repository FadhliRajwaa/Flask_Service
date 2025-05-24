from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['retinopathy_prediction']
predictions_collection = db['predictions']

# Load Retinopathy model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model-Retinopaty.h5')
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Retinopathy class names
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def prepare_image(img, target_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img = image.load_img(file, target_size=(224, 224))
        img_array = prepare_image(img)
        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])
        class_name = CLASS_NAMES[class_idx]
        confidence = float(np.max(preds[0]))
        
        # Save prediction to MongoDB
        prediction_record = {
            'class': class_name,
            'confidence': confidence,
            'filename': file.filename,
            'timestamp': datetime.datetime.now()
        }
        
        result = predictions_collection.insert_one(prediction_record)
        prediction_id = str(result.inserted_id)
        
        return jsonify({
            'id': prediction_id,
            'class': class_name, 
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
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

@app.route('/stats', methods=['GET'])
def get_stats():
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

@app.route('/test-model', methods=['GET'])
def test_model():
    """Endpoint to test if model is loaded correctly"""
    if model is not None:
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully',
            'model_summary': str(model.summary())
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Model failed to load'
        }), 500

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'retinopathy-api',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 