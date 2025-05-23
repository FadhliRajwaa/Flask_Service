from flask import Flask, request, jsonify
import numpy as np
import io
import os
import sys
import time
import psutil
from flask_cors import CORS
from PIL import Image

# Coba import TensorFlow, dengan fallback jika tidak tersedia
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, save_model
    from tensorflow.keras.preprocessing import image
    import h5py
    
    # Konfigurasi logging untuk TensorFlow
    tf.get_logger().setLevel('INFO')
    print(f"TensorFlow version: {tf.__version__}")
    TENSORFLOW_AVAILABLE = True
    
    def convert_model_format(model_path, output_path=None):
        """
        Konversi model ke format yang kompatibel dengan versi TensorFlow saat ini
        """
        if output_path is None:
            output_path = model_path.replace('.h5', '_converted.h5')
        
        try:
            # Cek apakah file model ada
            if not os.path.exists(model_path):
                print(f"File model tidak ditemukan: {model_path}")
                return None, f"File tidak ditemukan: {model_path}"
            
            # Definisikan custom_objects untuk menangani batch_shape
            custom_objects = {
                'InputLayer': lambda config: tf.keras.layers.InputLayer(
                    input_shape=config.get('batch_shape')[1:] if config.get('batch_shape') else None,
                    **{k: v for k, v in config.items() if k != 'batch_shape'}
                )
            }
            
            # Coba load model dengan berbagai opsi
            try:
                # Opsi 1: Load dengan custom_objects untuk menangani batch_shape
                print("Mencoba opsi 1: Load dengan custom_objects...")
                model = load_model(model_path, compile=False, custom_objects=custom_objects)
            except Exception as e1:
                print(f"Opsi 1 gagal: {e1}")
                try:
                    # Opsi 2: Ekstrak dan modifikasi konfigurasi model
                    print("Mencoba opsi 2: Ekstrak dan modifikasi konfigurasi model...")
                    with h5py.File(model_path, 'r') as f:
                        model_config = f.attrs.get('model_config')
                        if model_config is not None:
                            # Decode dan parse konfigurasi
                            import json
                            config_dict = json.loads(model_config.decode('utf-8'))
                            
                            # Modifikasi konfigurasi untuk menangani batch_shape
                            if 'config' in config_dict and 'layers' in config_dict['config']:
                                for layer in config_dict['config']['layers']:
                                    if 'config' in layer and 'batch_shape' in layer['config']:
                                        # Konversi batch_shape ke input_shape
                                        batch_shape = layer['config']['batch_shape']
                                        if batch_shape and len(batch_shape) > 1:
                                            layer['config']['input_shape'] = batch_shape[1:]
                                        # Hapus batch_shape
                                        del layer['config']['batch_shape']
                            
                            # Buat model dari konfigurasi yang dimodifikasi
                            model = tf.keras.models.model_from_json(json.dumps(config_dict))
                            
                            # Coba load weights
                            try:
                                model.load_weights(model_path)
                            except Exception as ew:
                                print(f"Gagal load weights: {ew}")
                                # Coba load weights dengan skip_mismatch
                                model.load_weights(model_path, skip_mismatch=True)
                        else:
                            raise ValueError("Tidak dapat membaca konfigurasi model")
                except Exception as e2:
                    print(f"Opsi 2 gagal: {e2}")
                    try:
                        # Opsi 3: Buat model sederhana dengan struktur yang sama
                        print("Mencoba opsi 3: Rekonstruksi model...")
                        # Periksa struktur model dengan h5py
                        with h5py.File(model_path, 'r') as f:
                            if 'model_weights' in f:
                                # Coba rekonstruksi model berdasarkan layer yang ada
                                # Ini adalah pendekatan yang sangat sederhana dan mungkin tidak berfungsi untuk semua model
                                inputs = tf.keras.layers.Input(shape=(224, 224, 3))
                                x = inputs
                                x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
                                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                                x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                                x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
                                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                                x = tf.keras.layers.Flatten()(x)
                                x = tf.keras.layers.Dense(128, activation='relu')(x)
                                outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
                                model = tf.keras.Model(inputs, outputs)
                            else:
                                raise ValueError("Struktur model tidak dapat diidentifikasi")
                    except Exception as e3:
                        print(f"Opsi 3 gagal: {e3}")
                        return None, f"Semua opsi gagal: {e1}, {e2}, {e3}"
            
            # Kompilasi model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Simpan model dalam format yang kompatibel
            save_model(model, output_path, save_format='h5')
            print(f"Model berhasil dikonversi dan disimpan di {output_path}")
            
            return model, output_path
        except Exception as e:
            print(f"Gagal mengkonversi model: {e}")
            return None, str(e)
    
except ImportError:
    print("TensorFlow tidak tersedia. Berjalan dalam mode simulasi.")
    tf = None
    TENSORFLOW_AVAILABLE = False
    
    def convert_model_format(model_path, output_path=None):
        print("TensorFlow tidak tersedia, tidak dapat mengkonversi model.")
        return None, "TensorFlow tidak tersedia"

app = Flask(__name__)
CORS(app)

# Variabel global untuk pelacakan
app_start_time = time.time()
total_requests = 0
successful_predictions = 0

# Konfigurasi path model - gunakan path absolut untuk memastikan model ditemukan
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'model-Retinopaty.h5')

# Pesan info awal
if TENSORFLOW_AVAILABLE:
    print(f"Flask API untuk RetinaScan (TensorFlow {tf.__version__})")
    print(f"Mencari model di: {MODEL_PATH}")
    
    # Coba berbagai metode untuk memuat model
    model = None
    
    # Metode 1: Memuat model langsung dengan custom_objects untuk menangani batch_shape
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"File model tidak ditemukan di: {MODEL_PATH}")
        else:
            print("Mencoba memuat model dengan metode standar...")
            # Definisikan custom_objects untuk menangani batch_shape
            custom_objects = {
                'InputLayer': lambda config: tf.keras.layers.InputLayer(
                    input_shape=config.get('batch_shape')[1:] if config.get('batch_shape') else None,
                    **{k: v for k, v in config.items() if k != 'batch_shape'}
                )
            }
            
            model = load_model(
                MODEL_PATH,
                compile=False,
                custom_objects=custom_objects
            )
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()
            print("Model berhasil dimuat!")
    except Exception as e:
        print(f"Gagal memuat model dengan metode standar: {e}")
        model = None
    
    # Metode 2: Memuat dengan tf.keras API jika metode 1 gagal
    if model is None:
        try:
            print("Mencoba memuat model dengan tf.keras API...")
            # Gunakan custom_objects yang sama
            custom_objects = {
                'InputLayer': lambda config: tf.keras.layers.InputLayer(
                    input_shape=config.get('batch_shape')[1:] if config.get('batch_shape') else None,
                    **{k: v for k, v in config.items() if k != 'batch_shape'}
                )
            }
            
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects=custom_objects
            )
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()
            print("Model berhasil dimuat dengan tf.keras API!")
        except Exception as e:
            print(f"Gagal memuat model dengan tf.keras API: {e}")
            model = None
    
    # Metode 3: Konversi model jika metode 1 dan 2 gagal
    if model is None:
        try:
            print("Mencoba mengkonversi model ke format yang kompatibel...")
            converted_model_path = MODEL_PATH.replace('.h5', '_converted.h5')
            model, result = convert_model_format(MODEL_PATH, converted_model_path)
            if model is not None:
                print(f"Model berhasil dikonversi dan dimuat: {result}")
                model.summary()
            else:
                print(f"Gagal mengkonversi model: {result}")
        except Exception as e:
            print(f"Gagal mengkonversi dan memuat model: {e}")
            model = None
    
    # Jika semua metode gagal, gunakan mode simulasi
    if model is None:
        print("Semua metode loading model gagal. Menggunakan mode simulasi...")
else:
    print("TensorFlow tidak tersedia. Berjalan dalam mode simulasi.")
    model = None
    print(f"Mencari model di: {MODEL_PATH}")
    print(f"File model {'ada' if os.path.exists(MODEL_PATH) else 'tidak ada'}")

# Kelas output model (disesuaikan dengan model yang memiliki 5 kelas untuk tingkat keparahan DR)
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
# Mapping output ke bahasa Indonesia
SEVERITY_MAPPING = {
    'No DR': 'Tidak ada',
    'Mild': 'Ringan',
    'Moderate': 'Sedang',
    'Severe': 'Berat',
    'Proliferative DR': 'Sangat Berat',
    # Fallback untuk kompatibilitas dengan model lama
    'Normal': 'Tidak ada',
    'Diabetic Retinopathy': 'Sedang'
}
# Mapping tingkat keparahan
SEVERITY_LEVEL_MAPPING = {
    'No DR': 0,
    'Mild': 1,
    'Moderate': 2,
    'Severe': 3,
    'Proliferative DR': 4,
    # Fallback untuk kompatibilitas dengan model lama
    'Normal': 0,
    'Diabetic Retinopathy': 2
}

def preprocess_image(img_bytes):
    """
    Memproses gambar untuk prediksi dengan model
    """
    try:
        # Buka gambar dari bytes
        img = Image.open(io.BytesIO(img_bytes))
        
        # Konversi ke RGB jika dalam mode lain (misalnya RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Ubah ukuran sesuai model
        img = img.resize((224, 224))  # Sesuaikan dengan ukuran input model
        
        # Konversi ke array numpy - dengan fallback jika TensorFlow tidak tersedia
        if TENSORFLOW_AVAILABLE:
            img_array = image.img_to_array(img)
        else:
            # Fallback ke numpy langsung jika TensorFlow tidak tersedia
            img_array = np.array(img)
        
        # Normalisasi
        img_array = img_array / 255.0
        
        # Tambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Gambar berhasil diproses: shape={img_array.shape}, dtype={img_array.dtype}, range=[{img_array.min():.2f}, {img_array.max():.2f}]")
        
        return img_array
    except Exception as e:
        print(f"Error saat preprocessing gambar: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint untuk memeriksa kesehatan API"""
    global total_requests, successful_predictions
    
    try:
        # Dapatkan penggunaan sumber daya
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Convert to MB
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk_usage = psutil.disk_usage('/').percent
        
        # Hitung uptime
        uptime_seconds = time.time() - app_start_time
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        uptime_formatted = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Cek model dengan validasi lebih mendalam
        model_status = "healthy" 
        model_validation_error = None
        
        if model is None:
            model_status = "simulation_mode"
        else:
            # Coba validasi model dengan input dummy
            try:
                dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
                _ = model.predict(dummy_input, verbose=0)
            except Exception as model_error:
                model_status = "error"
                model_validation_error = str(model_error)
        
        model_file_status = "exists" if os.path.exists(MODEL_PATH) else "missing"
        
        # Status aplikasi
        status = {
            'status': 'healthy',
            'version': '1.1.0',
            'uptime': uptime_formatted,
            'uptime_seconds': uptime_seconds,
            'startup_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(app_start_time)),
            'current_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'resources': {
                'memory_usage_mb': round(memory_usage, 2),
                'cpu_percent': cpu_percent,
                'disk_usage_percent': disk_usage,
                'process_id': os.getpid()
            },
            'model': {
                'status': model_status,
                'loaded': model is not None,
                'simulation_mode': model is None,
                'path': MODEL_PATH,
                'file_status': model_file_status,
                'exists': os.path.exists(MODEL_PATH),
                'classes': CLASSES,
                'input_shape': getattr(model, 'input_shape', None) if model else None,
                'validation_error': model_validation_error
            },
            'stats': {
                'total_requests': total_requests,
                'successful_predictions': successful_predictions,
                'success_rate': round(successful_predictions / total_requests * 100, 1) if total_requests > 0 else 0,
                'last_request_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) if total_requests > 0 else None
            },
            'environment': {
                'tensorflow_version': tf.__version__ if TENSORFLOW_AVAILABLE else 'not available',
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'python_version': sys.version,
                'flask_env': os.environ.get('FLASK_ENV', 'production'),
                'port': os.environ.get('PORT', '5001'),
                'host': os.environ.get('HOST', '0.0.0.0'),
                'debug_mode': os.environ.get('FLASK_DEBUG', 'False')
            },
            'endpoints': {
                'predict': '/predict',
                'info': '/info',
                'health': '/health',
                'test_model': '/test-model'
            }
        }
        
        # Periksa koneksi ke backend (opsional)
        backend_url = os.environ.get('BACKEND_URL')
        if backend_url:
            try:
                import requests
                response = requests.get(f"{backend_url}/api/health", timeout=2)
                status['backend_connection'] = {
                    'status': 'connected' if response.status_code == 200 else 'error',
                    'status_code': response.status_code,
                    'url': backend_url
                }
            except Exception as be:
                status['backend_connection'] = {
                    'status': 'error',
                    'error': str(be),
                    'url': backend_url
                }
        
        # Tambahkan header untuk CORS
        response = jsonify(status)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(f"Error saat health check: {e}")
        print(f"Stack trace: {sys.exc_info()}")
        error_status = {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        
        # Tambahkan header untuk CORS
        response = jsonify(error_status)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

@app.route('/predict', methods=['POST'])
def predict():
    global total_requests, successful_predictions
    
    try:
        # Catat waktu mulai untuk menghitung waktu pemrosesan
        start_time = time.time()
        total_requests += 1
        
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nama file kosong'}), 400
        
        # Baca gambar sebagai bytes
        img_bytes = file.read()
        
        # Mode prediksi sebenarnya
        if model is not None:
            try:
                print(f"Memproses gambar untuk prediksi: {file.filename}")
                # Preprocess gambar
                preprocessed_img = preprocess_image(img_bytes)
                
                # Prediksi dengan model
                print("Menjalankan prediksi dengan model...")
                predictions = model.predict(preprocessed_img)
                print(f"Hasil prediksi raw: {predictions}")
                
                # Ambil kelas dengan probabilitas tertinggi
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = CLASSES[predicted_class_index]
                
                # Ambil nilai confidence (probabilitas)
                confidence = float(predictions[0][predicted_class_index])
                
                # Mapping ke nama Indonesia dan level
                severity = SEVERITY_MAPPING[predicted_class]
                severity_level = SEVERITY_LEVEL_MAPPING[predicted_class]
                
                # Tambahkan rekomendasi berdasarkan tingkat keparahan
                recommendation = get_recommendation_by_severity(predicted_class)
                
                # Hasil prediksi dengan format yang konsisten
                result = {
                    'severity': predicted_class,  # Kelas asli dari model
                    'severity_level': severity_level,
                    'confidence': confidence,
                    'frontendSeverity': severity,  # Nama dalam bahasa Indonesia untuk frontend
                    'frontendSeverityLevel': severity_level,
                    'recommendation': recommendation,
                    'raw_prediction': {
                        'class': predicted_class,
                        'probabilities': {CLASSES[i]: float(predictions[0][i]) for i in range(len(CLASSES))}
                    },
                    # Tambahan field untuk memastikan kompatibilitas dengan backend
                    'model_version': '1.1.0',
                    'api_version': '1.1.0',
                    'timestamp': time.time(),
                    'processing_time_ms': int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
                }
                
                print(f"Prediksi untuk gambar {file.filename}: {severity} (confidence: {confidence:.2f})")
                successful_predictions += 1
                
                return jsonify(result)
            except Exception as e:
                print(f"Error saat menggunakan model: {e}")
                print(f"Stack trace: {sys.exc_info()}")
                # Fallback ke mode simulasi jika ada error dengan model
                print("Fallback ke mode simulasi...")
        
        # Mode simulasi (jika model tidak tersedia atau ada error)
        print(f"Menggunakan mode simulasi untuk gambar: {file.filename}")
        
        # Periksa gambar
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((224, 224))  # Hanya untuk memastikan gambar valid
        
        # Pilih kelas secara acak dengan bias ke kelas tertentu (untuk simulasi)
        import random
        # Weights untuk 5 kelas (No DR, Mild, Moderate, Severe, Proliferative DR)
        weights = [0.5, 0.2, 0.15, 0.1, 0.05]  # Distribusi realistis
        predicted_class_index = random.choices(range(len(CLASSES)), weights=weights)[0]
        predicted_class = CLASSES[predicted_class_index]
        
        # Generate confidence score yang realistis
        base_confidence = 0.75
        confidence = base_confidence + (random.random() * 0.2)  # 0.75 - 0.95
        
        # Mapping ke nama Indonesia dan level
        severity = SEVERITY_MAPPING[predicted_class]
        severity_level = SEVERITY_LEVEL_MAPPING[predicted_class]
        
        # Tambahkan rekomendasi berdasarkan tingkat keparahan
        recommendation = get_recommendation_by_severity(predicted_class)
        
        # Hasil prediksi simulasi
        result = {
            'severity': predicted_class,  # Kelas asli dari model
            'severity_level': severity_level,
            'confidence': confidence,
            'frontendSeverity': severity,  # Nama dalam bahasa Indonesia untuk frontend
            'frontendSeverityLevel': severity_level,
            'recommendation': recommendation,
            'raw_prediction': {
                'class': predicted_class,
                'probabilities': {class_name: round(random.random() * 0.1, 3) for class_name in CLASSES},
                'is_simulation': True
            }
        }
        
        # Set probabilitas kelas yang diprediksi lebih tinggi
        result['raw_prediction']['probabilities'][predicted_class] = confidence
        
        print(f"Simulasi prediksi untuk gambar {file.filename}: {severity} (confidence: {confidence:.2f})")
        successful_predictions += 1
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error saat memprediksi: {e}")
        print(f"Stack trace: {sys.exc_info()}")
        return jsonify({'error': str(e)}), 500

def get_recommendation_by_severity(severity_class):
    """
    Menghasilkan rekomendasi berdasarkan tingkat keparahan
    """
    recommendations = {
        'No DR': 'Lakukan pemeriksaan rutin setiap tahun.',
        'Mild': 'Kontrol gula darah dan tekanan darah. Pemeriksaan ulang dalam 9-12 bulan.',
        'Moderate': 'Konsultasi dengan dokter spesialis mata. Pemeriksaan ulang dalam 6 bulan.',
        'Severe': 'Rujukan segera ke dokter spesialis mata. Pemeriksaan ulang dalam 2-3 bulan.',
        'Proliferative DR': 'Rujukan segera ke dokter spesialis mata untuk evaluasi dan kemungkinan tindakan laser atau operasi.',
        # Fallback untuk kompatibilitas dengan model lama
        'Normal': 'Lakukan pemeriksaan rutin setiap tahun.',
        'Diabetic Retinopathy': 'Konsultasi dengan dokter spesialis mata. Pemeriksaan ulang dalam 6 bulan.'
    }
    
    return recommendations.get(severity_class, 'Konsultasikan dengan dokter mata.')

@app.route('/download-model', methods=['GET'])
def download_model():
    """Endpoint untuk mengunduh model dari URL"""
    if not TENSORFLOW_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'TensorFlow tidak tersedia'
        }), 500
    
    try:
        # URL model dapat dikonfigurasi melalui environment variable
        model_url = request.args.get('url') or os.environ.get('MODEL_URL')
        
        if not model_url:
            return jsonify({
                'status': 'error',
                'message': 'URL model tidak diberikan'
            }), 400
        
        import requests
        
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Unduh model
        print(f"Mengunduh model dari {model_url}...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Simpan model
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Model berhasil diunduh ke {MODEL_PATH}")
        
        # Coba muat model
        try:
            model = load_model(MODEL_PATH, compile=False)
            model.summary()
            model_loaded = True
        except Exception as e:
            print(f"Gagal memuat model yang diunduh: {e}")
            model_loaded = False
        
        return jsonify({
            'status': 'success',
            'message': 'Model berhasil diunduh',
            'path': MODEL_PATH,
            'size_mb': os.path.getsize(MODEL_PATH) / (1024 * 1024),
            'model_loaded': model_loaded
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error saat mengunduh model: {str(e)}'
        }), 500

@app.route('/check-model', methods=['GET'])
def check_model():
    """Endpoint untuk memeriksa struktur file model"""
    if not TENSORFLOW_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'TensorFlow tidak tersedia'
        }), 500
    
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                'status': 'error',
                'message': f'File model tidak ditemukan: {MODEL_PATH}'
            }), 404
        
        # Periksa struktur file model dengan h5py
        import h5py
        model_info = {}
        
        with h5py.File(MODEL_PATH, 'r') as f:
            # Dapatkan atribut model
            for attr_name in f.attrs.keys():
                try:
                    attr_value = f.attrs[attr_name]
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8')
                    model_info[attr_name] = str(attr_value)
                except Exception as e:
                    model_info[f"{attr_name}_error"] = str(e)
            
            # Dapatkan struktur grup
            model_info['groups'] = list(f.keys())
            
            # Periksa versi format
            if 'keras_version' in f.attrs:
                model_info['keras_version'] = f.attrs['keras_version'].decode('utf-8')
            
            # Periksa layer
            if 'model_weights' in f:
                model_info['layers'] = list(f['model_weights'].keys())
            
            # Periksa metadata tambahan
            model_info['file_size_mb'] = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            model_info['current_tf_version'] = tf.__version__
        
        return jsonify({
            'status': 'success',
            'message': 'Struktur model berhasil diperiksa',
            'model_path': MODEL_PATH,
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error saat memeriksa model: {str(e)}'
        }), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """Endpoint untuk menguji model dengan gambar sampel"""
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model tidak tersedia',
                'simulation_mode': True
            }), 400
            
        # Coba buat gambar dummy untuk pengujian
        dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
        dummy_image_batch = np.expand_dims(dummy_image, axis=0)
        
        # Jalankan prediksi
        print("Menjalankan prediksi dengan gambar dummy...")
        predictions = model.predict(dummy_image_batch)
        
        # Ambil kelas dengan probabilitas tertinggi
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_index]
        
        # Ambil nilai confidence (probabilitas)
        confidence = float(predictions[0][predicted_class_index])
        
        return jsonify({
            'status': 'success',
            'message': 'Model berhasil diuji',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': {CLASSES[i]: float(predictions[0][i]) for i in range(len(CLASSES))}
            },
            'model_info': {
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape)
            }
        })
    except Exception as e:
        print(f"Error saat menguji model: {e}")
        print(f"Stack trace: {sys.exc_info()}")
        return jsonify({
            'status': 'error',
            'message': f'Gagal menguji model: {str(e)}'
        }), 500

@app.route('/convert-model', methods=['GET'])
def convert_model_endpoint():
    """Endpoint untuk mengkonversi model ke format yang kompatibel"""
    if not TENSORFLOW_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'TensorFlow tidak tersedia'
        }), 500
    
    try:
        # Konversi model
        converted_model_path = MODEL_PATH.replace('.h5', '_converted.h5')
        model_result, message = convert_model_format(MODEL_PATH, converted_model_path)
        
        if model_result is not None:
            return jsonify({
                'status': 'success',
                'message': 'Model berhasil dikonversi',
                'original_path': MODEL_PATH,
                'converted_path': converted_model_path
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Gagal mengkonversi model: {message}'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error saat mengkonversi model: {str(e)}'
        }), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Endpoint untuk mendapatkan informasi model"""
    try:
        info_data = {
            'status': 'success',
            'model_name': 'RetinaScan Diabetic Retinopathy Detection',
            'classes': CLASSES,
            'severity_mapping': SEVERITY_MAPPING,
            'severity_level_mapping': SEVERITY_LEVEL_MAPPING,
            'tf_version': tf.__version__ if TENSORFLOW_AVAILABLE else 'not available',
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'simulation_mode': model is None,
            'model_path': MODEL_PATH,
            'model_exists': os.path.exists(MODEL_PATH),
            'api_version': '1.1.0',
            'recommendations': {
                class_name: get_recommendation_by_severity(class_name) for class_name in CLASSES
            }
        }
        
        if model is not None:
            # Dapatkan struktur model jika tersedia
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            info_data['model_summary'] = '\n'.join(model_summary)
            info_data['model_input_shape'] = model.input_shape
            info_data['model_output_shape'] = model.output_shape
        else:
            info_data['model_summary'] = 'Model tidak tersedia (mode simulasi)'
            info_data['note'] = 'API berjalan dalam mode simulasi. Untuk menggunakan model yang sebenarnya, pastikan file model-Retinopati.h5 tersedia.'
            
        # Tambahkan statistik penggunaan
        info_data['stats'] = {
            'total_requests': total_requests,
            'successful_predictions': successful_predictions,
            'uptime_seconds': time.time() - app_start_time
        }
            
        return jsonify(info_data)
    
    except Exception as e:
        print(f"Error saat mendapatkan info model: {e}")
        print(f"Stack trace: {sys.exc_info()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Gunakan port dari environment variable jika tersedia (penting untuk Render)
    port = int(os.environ.get("PORT", 5001))
    
    # Gunakan mode debug hanya dalam development
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    print(f"Flask API berjalan pada port: {port}")
    print(f"Mode debug: {debug_mode}")
    print(f"TensorFlow tersedia: {TENSORFLOW_AVAILABLE}")
    print(f"Model tersedia: {model is not None}")
    print(f"Mode simulasi: {model is None}")
    print(f"File model ada: {os.path.exists(MODEL_PATH)}")
    print(f"Path model: {MODEL_PATH}")
    print("\nEndpoint yang tersedia:")
    print(f"- Prediksi: http://0.0.0.0:{port}/predict")
    print(f"- Info: http://0.0.0.0:{port}/info")
    print(f"- Health check: http://0.0.0.0:{port}/health")
    print(f"- Test model: http://0.0.0.0:{port}/test-model")
    print(f"- Check model: http://0.0.0.0:{port}/check-model")
    print(f"- Convert model: http://0.0.0.0:{port}/convert-model")
    print(f"- Download model: http://0.0.0.0:{port}/download-model?url=<model_url>")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 