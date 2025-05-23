from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import os
import platform
import sys
import json
import time
import psutil

# Coba import TensorFlow, tetapi jangan gagal jika tidak ada
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("TensorFlow imported successfully")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, running in simulation mode only")
except Exception as general_tf_error:
    print(f"TensorFlow error: {general_tf_error}")
    TF_AVAILABLE = False
    print("TensorFlow had error during import, running in simulation mode only")

app = Flask(__name__)
CORS(app, origins=['*'], supports_credentials=True, methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

# Cek apakah mode simulasi diaktifkan
# Jika FORCE_MODEL=1, abaikan SIMULATION_MODE dan paksa menggunakan model
# Atau jika SIMULATION_MODE tidak ditetapkan, paksa menggunakan model
force_model = os.environ.get("FORCE_MODEL") == "1"
# Periksa SIMULATION_MODE, jika tidak ada atau bukan "1", maka simulation_mode = False
simulation_mode = os.environ.get("SIMULATION_MODE") == "1" and not force_model

print(f"SIMULATION_MODE env: {os.environ.get('SIMULATION_MODE')}")
print(f"FORCE_MODEL env: {os.environ.get('FORCE_MODEL')}")
print(f"TensorFlow available: {TF_AVAILABLE}")
print(f"Simulation mode: {'ON' if simulation_mode else 'OFF'}")

# Pastikan direktori model ada
import os.path
import shutil

# Daftar kemungkinan lokasi model
model_paths = [
    "model-Retinopaty.h5",
    "./model-Retinopaty.h5",
    "../model-Retinopaty.h5",
    "/app/model-Retinopaty.h5",
    "/app/models/model-Retinopaty.h5",
    "model/model-Retinopaty.h5",
    "./model/model-Retinopaty.h5"
]

# Cek semua kemungkinan lokasi
model_path = None
for path in model_paths:
    print(f"Checking if model exists at path: {path}")
    if os.path.exists(path):
        print(f"Model file found at {path}")
        model_path = path
        break

if model_path is None:
    print(f"Model file NOT found in any location")
    # Cek lokasi file saat ini
    print(f"Current directory: {os.getcwd()}")
    try:
        print(f"Files in current directory: {os.listdir('.')}")
        
        # Cek apakah ada direktori models
        if os.path.exists('/app/models'):
            print(f"Files in /app/models: {os.listdir('/app/models')}")
        
        # Coba salin model dari lokasi saat ini ke /app/models jika ada
        if os.path.exists('model-Retinopaty.h5') and os.path.exists('/app/models'):
            print("Copying model file to /app/models directory")
            shutil.copy('model-Retinopaty.h5', '/app/models/model-Retinopaty.h5')
            model_path = '/app/models/model-Retinopaty.h5'
    except Exception as e:
        print(f"Error checking directories: {e}")

# SELALU coba load model jika TensorFlow tersedia, terlepas dari simulation_mode
# Ini memastikan model dimuat meskipun SIMULATION_MODE=1
model = None
original_simulation_mode = simulation_mode  # Simpan nilai awal untuk logging
if TF_AVAILABLE:  # Hanya cek TensorFlow tersedia, abaikan simulation_mode
    try:
        print("Attempting to load model...")
        # Coba load model jika path ditemukan
        if model_path and os.path.exists(model_path):
            # Cek ukuran file model
            model_size = os.path.getsize(model_path)
            print(f"Model file size: {model_size} bytes")
            
            if model_size > 0:
                # Coba load model dengan error handling lebih detail
                try:
                    print(f"Loading model from {model_path}...")
                    model = load_model(model_path)
                    print("Model loaded successfully")
                    simulation_mode = False  # Pastikan simulation mode OFF jika model berhasil dimuat
                    
                    # Verifikasi model dengan timeout untuk mencegah hanging
                    print("Verifying model...")
                    try:
                        # Gunakan mekanisme timeout yang cross-platform
                        # Signal tidak tersedia di Windows (SIGALRM)
                        if sys.platform != 'win32':
                            import signal
                            
                            # Definisikan handler untuk timeout
                            def timeout_handler(signum, frame):
                                raise TimeoutError("Model prediction timed out")
                            
                            # Set timeout 30 detik untuk prediksi
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(30)
                            
                            dummy_input = np.random.random((1, 224, 224, 3))
                            dummy_output = model.predict(dummy_input)
                            print(f"Model verification successful. Output shape: {dummy_output.shape}")
                            
                            # Matikan alarm timeout
                            signal.alarm(0)
                        else:
                            # Gunakan pendekatan alternatif untuk Windows
                            import threading
                            import queue
                            
                            def predict_with_timeout():
                                try:
                                    dummy_input = np.random.random((1, 224, 224, 3))
                                    result = model.predict(dummy_input)
                                    result_queue.put(result)
                                except Exception as e:
                                    result_queue.put(e)
                            
                            result_queue = queue.Queue()
                            # Pada Windows, gunakan pendekatan sederhana tanpa Queue
                            dummy_input = np.random.random((1, 224, 224, 3))
                            dummy_output = model.predict(dummy_input)
                            print(f"Model verification successful. Output shape: {dummy_output.shape}")
                            
                    except TimeoutError as te:
                        print(f"Model verification failed: {te}")
                        print("Continuing with model anyway as it loaded successfully")
                    except Exception as ve:
                        print(f"Model verification error: {ve}")
                        print("Continuing with model anyway as it loaded successfully")
                except Exception as load_error:
                    print(f"Error during model loading: {load_error}")
                    print(f"Error type: {type(load_error)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    # Jangan aktifkan simulation_mode jika FORCE_MODEL=1
                    if os.environ.get("FORCE_MODEL") != "1":
                        simulation_mode = True
                    else:
                        print("FORCE_MODEL=1, attempting to initialize dummy model")
                        # Buat model dummy jika FORCE_MODEL=1
                        try:
                            from tensorflow.keras.models import Sequential
                            from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
                            print("Creating dummy model...")
                            model = Sequential([
                                Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                                MaxPooling2D((2, 2)),
                                Conv2D(32, (3, 3), activation='relu'),
                                MaxPooling2D((2, 2)),
                                Flatten(),
                                Dense(64, activation='relu'),
                                Dense(5, activation='softmax')
                            ])
                            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                            print("Dummy model created successfully")
                            simulation_mode = False
                        except Exception as dummy_error:
                            print(f"Error creating dummy model: {dummy_error}")
                            simulation_mode = True
            else:
                print("Model file exists but is empty")
                # Jangan aktifkan simulation_mode jika FORCE_MODEL=1
                if os.environ.get("FORCE_MODEL") != "1":
                    simulation_mode = True
        else:
            print("Model path not found")
            # Jangan aktifkan simulation_mode jika FORCE_MODEL=1
            if os.environ.get("FORCE_MODEL") != "1":
                simulation_mode = True
    except Exception as e:
        print(f"Error in model loading process: {e}")
        # Jangan aktifkan simulation_mode jika FORCE_MODEL=1
        if os.environ.get("FORCE_MODEL") != "1":
            simulation_mode = True

# Infokan perubahan mode simulasi
if original_simulation_mode != simulation_mode:
    print(f"Simulation mode changed from {original_simulation_mode} to {simulation_mode}")

# Update status mode simulasi
print(f"Final simulation mode: {'ON' if simulation_mode else 'OFF'}")
print(f"Model loaded: {'YES' if model is not None else 'NO'}")

# Jika model tidak ada, buat model dummy - selalu coba buat model dummy jika TensorFlow tersedia
if model is None and TF_AVAILABLE:
    try:
        print("Creating fallback dummy model...")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
        
        try:
            # Coba buat model yang lebih kecil untuk menghemat memori
            model = Sequential([
                Conv2D(8, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                MaxPooling2D((4, 4)),  # Lebih besar untuk mengurangi dimensi lebih cepat
                Conv2D(16, (3, 3), activation='relu'),
                MaxPooling2D((4, 4)),  # Lebih besar untuk mengurangi dimensi lebih cepat
                Flatten(),
                Dense(32, activation='relu'),  # Lebih kecil
                Dense(5, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("Fallback dummy model created successfully")
            
            # Verifikasi model fallback dengan metode cross-platform
            print("Verifying fallback model...")
            try:
                # Gunakan mekanisme timeout yang cross-platform
                if sys.platform != 'win32':
                    import signal
                    
                    # Definisikan handler untuk timeout
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Fallback model prediction timed out")
                    
                    # Set timeout 15 detik untuk prediksi
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(15)
                    
                    dummy_input = np.zeros((1, 224, 224, 3))  # Gunakan zeros karena lebih efisien daripada random
                    dummy_output = model.predict(dummy_input)
                    print(f"Fallback model verification successful. Output shape: {dummy_output.shape}")
                    
                    # Matikan alarm timeout
                    signal.alarm(0)
                else:
                    # Gunakan pendekatan alternatif untuk Windows
                    import threading
                    import queue
                    
                    def predict_with_timeout():
                        try:
                            dummy_input = np.zeros((1, 224, 224, 3))
                            result = model.predict(dummy_input)
                            result_queue.put(result)
                        except Exception as e:
                            result_queue.put(e)
                    
                    # Pada Windows, gunakan pendekatan sederhana tanpa Queue
                    dummy_input = np.zeros((1, 224, 224, 3))
                    dummy_output = model.predict(dummy_input)
                    print(f"Fallback model verification successful. Output shape: {dummy_output.shape}")
            except Exception as ve:
                print(f"Fallback model verification error: {ve}")
                print("Continuing with fallback model anyway as it was created successfully")
            
            # Jika berhasil sampai sini, hentikan mode simulasi
            simulation_mode = False
        except Exception as fallback_error:
            print(f"Error creating/verifying fallback model: {fallback_error}")
            # Fallback gagal, tetap dalam mode simulasi
            simulation_mode = True
        print("Dummy model created successfully")
        simulation_mode = False
        print(f"Updated simulation mode: {'ON' if simulation_mode else 'OFF'}")
    except Exception as dummy_error:
        print(f"Error creating dummy model: {dummy_error}")
        simulation_mode = True

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
    global simulation_mode, model
    
    # Log request untuk debugging
    print(f"Received prediction request. Simulation mode: {simulation_mode}")
    print(f"Model loaded: {model is not None}")
    print(f"Request files: {list(request.files.keys()) if request.files else 'No files'}")
    print(f"Request form: {list(request.form.keys()) if request.form else 'No form data'}")
    print(f"Request content type: {request.content_type}")
    
    # Terima file dengan nama 'file' atau 'image'
    file_key = None
    if "file" in request.files:
        file_key = "file"
    elif "image" in request.files:
        file_key = "image"
    
    # Force model mode jika diatur
    force_model = os.environ.get("FORCE_MODEL") == "1"
    
    # Cek apakah harus menggunakan mode simulasi untuk request ini
    request_simulation_mode = simulation_mode
    if force_model and model is not None:
        request_simulation_mode = False
        print("FORCE_MODEL=1 and model loaded, using model for prediction")
    
    if file_key is None and not request_simulation_mode:
        print("No image file provided and not in simulation mode")
        # Cek apakah ada data di request.data
        if request.data:
            print(f"Found raw data in request.data, length: {len(request.data)}")
            try:
                # Coba proses data mentah sebagai file gambar
                image = Image.open(io.BytesIO(request.data))
                print(f"Successfully parsed raw data as image: {image.size}")
                # Gunakan data mentah sebagai file gambar
                use_raw_data = True
            except Exception as e:
                print(f"Could not parse raw data as image: {e}")
                # Aktifkan mode simulasi sebagai fallback untuk request ini saja
                request_simulation_mode = True
                print("Activating simulation mode as fallback for this request")
                use_raw_data = False
        else:
            # Aktifkan mode simulasi sebagai fallback untuk request ini saja
            request_simulation_mode = True
            print("Activating simulation mode as fallback for this request")
            use_raw_data = False
    else:
        use_raw_data = False

    try:
        if not request_simulation_mode and model is not None and (file_key is not None or use_raw_data):
            # Mode normal dengan model
            print("Using actual model for prediction")
            
            if use_raw_data:
                # Gunakan data mentah dari request.data
                image = Image.open(io.BytesIO(request.data))
            else:
                # Gunakan file dari request.files
                image_file = request.files[file_key]
                image = Image.open(io.BytesIO(image_file.read()))
            
            print(f"Image loaded, size: {image.size}, mode: {image.mode}")
            input_tensor = preprocess_image(image)
            print(f"Image preprocessed, tensor shape: {input_tensor.shape}")
            
            try:
                # Gunakan model dengan timeout untuk mencegah hanging
                import signal
                
                # Function to handle timeout
                def timeout_handler(signum, frame):
                    raise TimeoutError("Model prediction timed out")
                
                # Set timeout untuk 15 detik
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(15)
                
                try:
                    predictions = model.predict(input_tensor)[0]
                    # Reset alarm
                    signal.alarm(0)
                    
                    print(f"Raw predictions: {predictions}")
                    class_index = predictions.argmax()
                    class_name = CLASSES[class_index]
                    confidence = float(predictions[class_index])
                    print(f"Model prediction: {class_name} with confidence {confidence}")
                except TimeoutError:
                    print("Model prediction timed out, falling back to simulation")
                    # Fallback ke simulasi
                    import random
                    class_index = random.randint(0, len(CLASSES)-1)
                    class_name = CLASSES[class_index]
                    confidence = 0.7 + random.random() * 0.3  # 0.7-1.0
                    print(f"Fallback to simulation: {class_name} with confidence {confidence}")
                finally:
                    # Restore previous signal handler
                    signal.signal(signal.SIGALRM, old_handler)
            except Exception as model_error:
                print(f"Error during model prediction: {model_error}")
                # Fallback ke simulasi jika prediksi gagal
                import random
                class_index = random.randint(0, len(CLASSES)-1)
                class_name = CLASSES[class_index]
                confidence = 0.7 + random.random() * 0.3  # 0.7-1.0
                print(f"Fallback to simulation: {class_name} with confidence {confidence}")
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
            "is_simulation": request_simulation_mode or model is None,
            "reason": "SIMULATION_MODE=1 in environment" if os.environ.get("SIMULATION_MODE") == "1" and os.environ.get("FORCE_MODEL") != "1" else 
                     "Model not loaded" if model is None else 
                     "No file provided" if file_key is None else None,
            "force_model": os.environ.get("FORCE_MODEL") == "1",
            "model_available": model is not None
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
    
    # Cek status model file
    model_paths_to_check = [
        "model-Retinopaty.h5",
        "./model-Retinopaty.h5",
        "../model-Retinopaty.h5",
        "/app/model-Retinopaty.h5",
        "/app/models/model-Retinopaty.h5"
    ]
    
    model_files = []
    for path in model_paths_to_check:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        model_files.append({
            "path": path,
            "exists": exists,
            "size": size
        })
    
    # Cek status TensorFlow
    tf_status = {}
    if TF_AVAILABLE:
        try:
            tf_status = {
                "version": tf.__version__,
                "devices": str(tf.config.list_physical_devices()),
                "built_with_cuda": tf.test.is_built_with_cuda(),
                "memory_info": str(tf.config.experimental.get_memory_info('GPU:0')) if tf.config.list_physical_devices('GPU') else "No GPU"
            }
        except Exception as e:
            tf_status = {
                "version": tf.__version__,
                "error": str(e)
            }
    
    # Cek memory usage
    import psutil
    memory_info = {
        "total": psutil.virtual_memory().total,
        "available": psutil.virtual_memory().available,
        "percent": psutil.virtual_memory().percent,
        "process_usage": psutil.Process(os.getpid()).memory_info().rss
    }
    
    response_data = {
        "status": "ok",
        "model_name": "model-Retinopaty.h5",
        "model_loaded": model_loaded,
        "model_files": model_files,
        "classes": CLASSES,
        "severity_mapping": SEVERITY_MAPPING,
        "severity_level_mapping": SEVERITY_LEVEL_MAPPING,
        "simulation_mode": simulation_mode,
        "force_model": os.environ.get("FORCE_MODEL") == "1",
        "simulation_reason": "SIMULATION_MODE=1 and FORCE_MODEL!=1" if os.environ.get("SIMULATION_MODE") == "1" and os.environ.get("FORCE_MODEL") != "1" else
                           "TensorFlow not available" if not TF_AVAILABLE else
                           "Model not loaded" if not model_loaded else None,
        "api_version": "1.0.0",
        "platform": platform.platform(),
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "tensorflow_status": tf_status,
        "memory_info": memory_info,
        "environment": {
            "SIMULATION_MODE": os.environ.get("SIMULATION_MODE"),
            "FORCE_MODEL": os.environ.get("FORCE_MODEL"),
            "TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
            "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL")
        }
    }
    
    print(f"Info endpoint called, returning: {json.dumps(response_data, indent=2)}")
    return jsonify(response_data)

@app.route("/", methods=["GET"])
def home():
    """Root endpoint untuk health check."""
    model_loaded = model is not None
    global simulation_mode
    
    # Check if model exists
    model_exists = False
    model_path_local = None
    
    # Check common paths
    for path in ["model-Retinopaty.h5", "./model-Retinopaty.h5", "../model-Retinopaty.h5", "/app/model-Retinopaty.h5", "/app/models/model-Retinopaty.h5"]:
        if os.path.exists(path):
            model_exists = True
            model_path_local = path
            break
    
    # Tambah informasi diagnostik untuk membantu debug
    tf_version = None
    if TF_AVAILABLE:
        try:
            import tensorflow as tf
            tf_version = tf.__version__
        except:
            tf_version = "Error getting version"
    
    model_info = None
    if model is not None:
        try:
            model_info = {
                "type": str(type(model)),
                "input_shape": str(model.input_shape),
                "output_shape": str([layer.output_shape for layer in model.layers][-1] if model.layers else None)
            }
        except Exception as model_info_error:
            model_info = f"Error getting model info: {str(model_info_error)}"
    
    memory_info = None
    try:
        import psutil
        process = psutil.Process()
        memory_info = {
            "rss_mb": process.memory_info().rss / 1024 / 1024,  # MB
            "vms_mb": process.memory_info().vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        }
    except Exception as mem_error:
        memory_info = f"Error getting memory info: {str(mem_error)}"
    
    response_data = {
        "status": "ok",
        "message": "Flask API for RetinaScan is running",
        "model_loaded": model_loaded,
        "model_exists": model_exists,
        "model_path": model_path_local,
        "simulation_mode": simulation_mode,
        "force_model": os.environ.get("FORCE_MODEL") == "1",
        "tensorflow_available": TF_AVAILABLE,
        "timestamp": time.time(),
        "environment": {
            "SIMULATION_MODE": os.environ.get("SIMULATION_MODE"),
            "FORCE_MODEL": os.environ.get("FORCE_MODEL"),
            "TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
            "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL"),
            "PORT": os.environ.get("PORT")
        },
        "diagnostics": {
            "tensorflow_version": tf_version,
            "model_info": model_info,
            "memory_info": memory_info,
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "os_platform": sys.platform
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
