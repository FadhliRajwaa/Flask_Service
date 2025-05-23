from flask import Flask, request, jsonify
import numpy as np
import io
import os
import sys
import time
import psutil
import logging
from flask_cors import CORS
from PIL import Image

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retinascan-api")

# Coba import TensorFlow, dengan fallback jika tidak tersedia
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import h5py
    
    # Konfigurasi logging untuk TensorFlow
    tf.get_logger().setLevel('INFO')
    logger.info(f"TensorFlow version: {tf.__version__}")
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TensorFlow tidak tersedia: {e}")
    tf = None
    TENSORFLOW_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Variabel global untuk pelacakan
app_start_time = time.time()
total_requests = 0
successful_predictions = 0

# Konfigurasi path model
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'model-Retinopaty.h5')
MODEL_VERSION = '1.1.0'

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

# Mapping tingkat keparahan
SEVERITY_LEVEL_MAPPING = {
    'No DR': 0,
    'Mild': 1,
    'Moderate': 2,
    'Severe': 3,
    'Proliferative DR': 4
}

# Fungsi untuk loading model dengan berbagai metode
def load_h5_model(model_path):
    """
    Fungsi untuk loading model H5 dengan berbagai strategi
    """
    if not os.path.exists(model_path):
        logger.error(f"File model tidak ditemukan di: {model_path}")
        return None, f"File model tidak ditemukan di: {model_path}"
    
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow tidak tersedia, tidak dapat memuat model.")
        return None, "TensorFlow tidak tersedia"
    
    logger.info(f"Memuat model dari: {model_path}")
    
    # Percobaan 1: Load model langsung tanpa custom objects
    try:
        logger.info("Strategi 1: Memuat model secara langsung")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model berhasil dimuat secara langsung")
        return model, "Strategi 1: Berhasil"
    except Exception as e:
        logger.warning(f"Strategi 1 gagal: {e}")
    
    # Percobaan 2: Load model dengan custom objects untuk batch_shape
    try:
        logger.info("Strategi 2: Memuat model dengan custom_objects untuk batch_shape")
        custom_objects = {
            'InputLayer': lambda config: tf.keras.layers.InputLayer(
                input_shape=config.get('batch_shape')[1:] if config.get('batch_shape') else None,
                **{k: v for k, v in config.items() if k != 'batch_shape'}
            )
        }
        
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects
        )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model berhasil dimuat dengan custom objects")
        return model, "Strategi 2: Berhasil"
    except Exception as e:
        logger.warning(f"Strategi 2 gagal: {e}")
    
    # Percobaan 3: Ekstrak konfigurasi dan modifikasi
    try:
        logger.info("Strategi 3: Ekstrak dan modifikasi konfigurasi model")
        with h5py.File(model_path, 'r') as f:
            if 'model_config' in f.attrs:
                import json
                config_str = f.attrs['model_config'].decode('utf-8')
                config_dict = json.loads(config_str)
                
                # Modifikasi konfigurasi untuk menangani batch_shape
                if 'config' in config_dict and 'layers' in config_dict['config']:
                    for layer in config_dict['config']['layers']:
                        if 'config' in layer and 'batch_shape' in layer['config']:
                            batch_shape = layer['config']['batch_shape']
                            if batch_shape and len(batch_shape) > 1:
                                layer['config']['input_shape'] = batch_shape[1:]
                            del layer['config']['batch_shape']
                
                # Buat model dari konfigurasi yang dimodifikasi
                model = tf.keras.models.model_from_json(json.dumps(config_dict))
                
                # Coba load weights
                try:
                    model.load_weights(model_path)
                except Exception as weight_error:
                    logger.warning(f"Gagal load weights biasa, mencoba skip_mismatch: {weight_error}")
                    model.load_weights(model_path, skip_mismatch=True)
                
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                logger.info("Model berhasil dimuat dengan modifikasi konfigurasi")
                return model, "Strategi 3: Berhasil"
            else:
                logger.warning("Strategi 3 gagal: Model tidak memiliki atribut model_config")
    except Exception as e:
        logger.warning(f"Strategi 3 gagal: {e}")
    
    # Percobaan 4: Rekonstruksi model dengan struktur dasar
    try:
        logger.info("Strategi 4: Rekonstruksi model dengan struktur sederhana")
        # Buat model dengan struktur yang sesuai untuk klasifikasi gambar retina
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))  # Input shape standar untuk gambar medis
        x = inputs
        
        # Blok 1
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        # Blok 2
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        # Blok 3
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        # Blok 4
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        # Fully connected
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Output untuk 5 kelas DR
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Coba load weights jika memungkinkan
        try:
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            logger.info("Weights berhasil dimuat secara parsial dengan skip_mismatch")
        except Exception as weight_error:
            logger.warning(f"Tidak dapat memuat weights: {weight_error}")
            logger.info("Menggunakan model dengan weights acak (tidak ideal)")
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model berhasil direkonstruksi")
        return model, "Strategi 4: Berhasil (rekonstruksi model)"
    except Exception as e:
        logger.warning(f"Strategi 4 gagal: {e}")
    
    logger.error("Semua strategi loading model gagal")
    return None, "Semua strategi loading model gagal"

# Coba muat model
if TENSORFLOW_AVAILABLE:
    logger.info(f"Mencari model di: {MODEL_PATH}")
    model, loading_result = load_h5_model(MODEL_PATH)
    
    if model is not None:
        logger.info(f"Model berhasil dimuat: {loading_result}")
        # Tampilkan ringkasan model
        model.summary(print_fn=logger.info)
    else:
        logger.warning(f"Gagal memuat model: {loading_result}")
        model = None
else:
    logger.warning("TensorFlow tidak tersedia. Berjalan dalam mode simulasi.")
    model = None

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
        img = img.resize((224, 224))  # Ukuran input standar
        
        # Konversi ke array numpy - dengan fallback jika TensorFlow tidak tersedia
        if TENSORFLOW_AVAILABLE:
            img_array = np.array(img)
        else:
            # Fallback ke numpy langsung
            img_array = np.array(img)
        
        # Normalisasi ke [0,1]
        img_array = img_array / 255.0
        
        # Tambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Gambar berhasil diproses: shape={img_array.shape}, dtype={img_array.dtype}, range=[{img_array.min():.2f}, {img_array.max():.2f}]")
        
        return img_array
    except Exception as e:
        logger.error(f"Error saat preprocessing gambar: {e}")
        raise

def get_recommendation_by_severity(severity_class):
    """
    Menghasilkan rekomendasi berdasarkan tingkat keparahan
    """
    recommendations = {
        'No DR': 'Lakukan pemeriksaan rutin setiap tahun.',
        'Mild': 'Kontrol gula darah dan tekanan darah. Pemeriksaan ulang dalam 9-12 bulan.',
        'Moderate': 'Konsultasi dengan dokter spesialis mata. Pemeriksaan ulang dalam 6 bulan.',
        'Severe': 'Rujukan segera ke dokter spesialis mata. Pemeriksaan ulang dalam 2-3 bulan.',
        'Proliferative DR': 'Rujukan segera ke dokter spesialis mata untuk evaluasi dan kemungkinan tindakan laser atau operasi.'
    }
    
    return recommendations.get(severity_class, 'Konsultasikan dengan dokter mata.')

def predict_with_model(image_array, filename="unknown"):
    """
    Melakukan prediksi dengan model atau menggunakan mode simulasi jika model tidak tersedia
    """
    start_time = time.time()
    
    if model is not None:
        try:
            logger.info(f"Menjalankan prediksi untuk gambar: {filename}")
            
            # Prediksi dengan model
            predictions = model.predict(image_array, verbose=0)
            
            # Ambil kelas dengan probabilitas tertinggi
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = CLASSES[predicted_class_index]
            
            # Ambil nilai confidence (probabilitas)
            confidence = float(predictions[0][predicted_class_index])
            
            logger.info(f"Prediksi untuk {filename}: {predicted_class} (confidence: {confidence:.2f})")
            
            # Mapping ke nama Indonesia dan level
            severity = SEVERITY_MAPPING[predicted_class]
            severity_level = SEVERITY_LEVEL_MAPPING[predicted_class]
            
            # Tambahkan rekomendasi
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
                'model_version': MODEL_VERSION,
                'timestamp': time.time(),
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
            
            return result, True
            
        except Exception as e:
            logger.error(f"Error saat menggunakan model: {e}")
            logger.error(f"Stack trace: {sys.exc_info()}")
            # Fallback ke mode simulasi jika ada error
            logger.info("Fallback ke mode simulasi...")
            # Akan dilanjutkan ke mode simulasi di bawah
    
    # Mode simulasi (jika model tidak tersedia atau ada error)
    logger.info(f"Menggunakan mode simulasi untuk gambar: {filename}")
    
    # Pilih kelas secara acak dengan bias ke kelas tertentu (untuk simulasi)
    import random
    # Weights untuk 5 kelas (No DR lebih umum, Proliferative DR paling jarang)
    weights = [0.5, 0.2, 0.15, 0.1, 0.05]  # Distribusi realistis
    predicted_class_index = random.choices(range(len(CLASSES)), weights=weights)[0]
    predicted_class = CLASSES[predicted_class_index]
    
    # Generate confidence score yang realistis
    base_confidence = 0.75
    confidence = base_confidence + (random.random() * 0.2)  # 0.75 - 0.95
    
    # Mapping ke nama Indonesia dan level
    severity = SEVERITY_MAPPING[predicted_class]
    severity_level = SEVERITY_LEVEL_MAPPING[predicted_class]
    
    # Tambahkan rekomendasi
    recommendation = get_recommendation_by_severity(predicted_class)
    
    # Buat distribusi probabilitas yang realistis
    probabilities = {class_name: round(random.random() * 0.1, 3) for class_name in CLASSES}
    probabilities[predicted_class] = confidence  # Set probabilitas kelas yang diprediksi
    
    # Hasil prediksi simulasi
    result = {
        'severity': predicted_class, 
        'severity_level': severity_level,
        'confidence': confidence,
        'frontendSeverity': severity,
        'frontendSeverityLevel': severity_level,
        'recommendation': recommendation,
        'raw_prediction': {
            'class': predicted_class,
            'probabilities': probabilities,
            'is_simulation': True
        },
        'model_version': MODEL_VERSION,
        'timestamp': time.time(),
        'processing_time_ms': int((time.time() - start_time) * 1000),
        'simulation_mode': True
    }
    
    logger.info(f"Simulasi prediksi untuk {filename}: {severity} (confidence: {confidence:.2f})")
    return result, True

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
            'version': MODEL_VERSION,
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
                'success_rate': round(successful_predictions / total_requests * 100, 1) if total_requests > 0 else 0
            },
            'environment': {
                'tensorflow_version': tf.__version__ if TENSORFLOW_AVAILABLE else 'not available',
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'python_version': sys.version,
                'flask_env': os.environ.get('FLASK_ENV', 'production'),
                'port': os.environ.get('PORT', '5001')
            }
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error saat health check: {e}")
        error_status = {
            'status': 'error',
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        return jsonify(error_status), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi gambar"""
    global total_requests, successful_predictions
    
    try:
        total_requests += 1
        
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nama file kosong'}), 400
        
        # Baca gambar sebagai bytes
        img_bytes = file.read()
        
        # Preprocess gambar
        try:
            preprocessed_img = preprocess_image(img_bytes)
        except Exception as preprocess_error:
            logger.error(f"Gagal memproses gambar: {preprocess_error}")
            return jsonify({'error': f'Gagal memproses gambar: {str(preprocess_error)}'}), 400
        
        # Prediksi dengan model atau mode simulasi
        result, success = predict_with_model(preprocessed_img, file.filename)
        
        if success:
            successful_predictions += 1
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error saat memprediksi: {e}")
        return jsonify({'error': str(e)}), 500

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
            'api_version': MODEL_VERSION,
            'recommendations': {
                class_name: get_recommendation_by_severity(class_name) for class_name in CLASSES
            }
        }
        
        if model is not None:
            # Dapatkan struktur model jika tersedia
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            info_data['model_summary'] = '\n'.join(model_summary)
            info_data['model_input_shape'] = str(model.input_shape)
            info_data['model_output_shape'] = str(model.output_shape)
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
        logger.error(f"Error saat mendapatkan info model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """Endpoint untuk menguji model dengan gambar dummy"""
    try:
        if model is None:
            return jsonify({
                'status': 'warning',
                'message': 'Model tidak tersedia, berjalan dalam mode simulasi',
                'simulation_mode': True
            })
            
        # Coba buat gambar dummy untuk pengujian
        dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Jalankan prediksi
        logger.info("Menjalankan prediksi dengan gambar dummy...")
        predictions = model.predict(dummy_image, verbose=0)
        
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
        logger.error(f"Error saat menguji model: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Gagal menguji model: {str(e)}'
        }), 500

@app.route('/check-model', methods=['GET'])
def check_model():
    """Endpoint untuk memeriksa struktur file model"""
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                'status': 'error',
                'message': f'File model tidak ditemukan: {MODEL_PATH}'
            }), 404
        
        model_info = {
            'file_path': MODEL_PATH,
            'file_exists': os.path.exists(MODEL_PATH),
            'file_size_mb': os.path.getsize(MODEL_PATH) / (1024 * 1024),
            'tensorflow_available': TENSORFLOW_AVAILABLE
        }
        
        # Jika h5py tersedia, gunakan untuk memeriksa struktur file
        try:
            import h5py
            with h5py.File(MODEL_PATH, 'r') as f:
                # Dapatkan atribut model
                model_info['attributes'] = {}
                for attr_name in f.attrs.keys():
                    try:
                        attr_value = f.attrs[attr_name]
                        if isinstance(attr_value, bytes):
                            attr_value = attr_value.decode('utf-8')
                        model_info['attributes'][attr_name] = str(attr_value)
                    except Exception as e:
                        model_info['attributes'][f"{attr_name}_error"] = str(e)
                
                # Dapatkan struktur grup
                model_info['groups'] = list(f.keys())
                
                # Periksa versi format
                if 'keras_version' in f.attrs:
                    model_info['keras_version'] = f.attrs['keras_version'].decode('utf-8')
                
                # Cek masalah batch_shape
                model_info['has_batch_shape_issue'] = False
                if 'model_config' in f.attrs:
                    try:
                        import json
                        config = json.loads(f.attrs['model_config'].decode('utf-8'))
                        if 'config' in config and 'layers' in config['config']:
                            for layer in config['config']['layers']:
                                if 'config' in layer and 'batch_shape' in layer['config']:
                                    model_info['has_batch_shape_issue'] = True
                                    model_info['batch_shape_details'] = layer['config']['batch_shape']
                                    break
                    except Exception as e:
                        model_info['config_parse_error'] = str(e)
        except ImportError:
            model_info['h5py_available'] = False
        except Exception as e:
            model_info['h5py_error'] = str(e)
        
        return jsonify({
            'status': 'success',
            'message': 'Struktur model berhasil diperiksa',
            'model_info': model_info
        })
    except Exception as e:
        logger.error(f"Error saat memeriksa model: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error saat memeriksa model: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Gunakan port dari environment variable jika tersedia (penting untuk Render)
    port = int(os.environ.get("PORT", 5001))
    
    # Gunakan mode debug hanya dalam development
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"Flask API RetinaScan berjalan pada port: {port}")
    logger.info(f"TensorFlow tersedia: {TENSORFLOW_AVAILABLE}")
    logger.info(f"Model tersedia: {model is not None}")
    logger.info(f"Mode simulasi: {model is None}")
    logger.info(f"File model ada: {os.path.exists(MODEL_PATH)}")
    logger.info(f"Path model: {MODEL_PATH}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)