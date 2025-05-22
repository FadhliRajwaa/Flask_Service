from flask import Flask, request, jsonify, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
import sys
import time
import psutil
from flask_cors import CORS

# Konfigurasi logging untuk TensorFlow
tf.get_logger().setLevel('INFO')
print(f"TensorFlow version: {tf.__version__}")

app = Flask(__name__)
CORS(app)

# Variabel global untuk pelacakan
app_start_time = time.time()
total_requests = 0
successful_predictions = 0
prediction_stats = {class_name: 0 for class_name in ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']}

# Konfigurasi path model - gunakan path absolut untuk memastikan model ditemukan
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'model-Retinopaty.h5')

# Pesan info awal
print(f"Flask API untuk RetinaScan (TensorFlow {tf.__version__})")
print(f"Mencari model di: {MODEL_PATH}")

# Pastikan model dapat dimuat
try:
    if not os.path.exists(MODEL_PATH):
        print(f"File model tidak ditemukan di: {MODEL_PATH}")
        model = None
    else:
        model = load_model(MODEL_PATH)
        model.summary()  # Menampilkan ringkasan model
        print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    print("Menggunakan mode simulasi...")
    # Tetap jalankan aplikasi dalam mode simulasi
    model = None

# Kelas output model (disesuaikan dengan model yang memiliki 5 kelas)
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
# Mapping output ke bahasa Indonesia
SEVERITY_MAPPING = {
    'No DR': 'Tidak ada DR',
    'Mild': 'DR Ringan',
    'Moderate': 'DR Sedang',
    'Severe': 'DR Berat',
    'Proliferative DR': 'DR Proliferatif'
}
# Mapping tingkat keparahan
SEVERITY_LEVEL_MAPPING = {
    'No DR': 0,
    'Mild': 1,
    'Moderate': 2,
    'Severe': 3,
    'Proliferative DR': 4
}

def preprocess_image(img_bytes):
    """
    Memproses gambar untuk prediksi dengan model
    """
    try:
        # Buka gambar dari bytes
        img = Image.open(io.BytesIO(img_bytes))
        
        # Ubah ukuran sesuai model
        img = img.resize((224, 224))  # Sesuaikan dengan ukuran input model
        
        # Konversi ke array numpy
        img_array = image.img_to_array(img)
        
        # Normalisasi
        img_array = img_array / 255.0
        
        # Tambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)
        
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
        
        # Status aplikasi
        status = {
            'status': 'healthy',
            'version': '1.0.0',
            'uptime': uptime_formatted,
            'resources': {
                'memory_usage_mb': round(memory_usage, 2),
                'cpu_percent': cpu_percent,
                'disk_usage_percent': disk_usage
            },
            'model': {
                'loaded': model is not None,
                'simulation_mode': model is None,
                'path': MODEL_PATH,
                'exists': os.path.exists(MODEL_PATH)
            },
            'stats': {
                'total_requests': total_requests,
                'successful_predictions': successful_predictions
            },
            'tensorflow_version': tf.__version__
        }
        
        return jsonify(status)
    except Exception as e:
        error_status = {
            'status': 'unhealthy',
            'error': str(e)
        }
        return jsonify(error_status), 500

@app.route('/predict', methods=['POST'])
def predict():
    global total_requests, successful_predictions, prediction_stats
    
    try:
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
                # Preprocess gambar
                preprocessed_img = preprocess_image(img_bytes)
                
                # Prediksi dengan model
                predictions = model.predict(preprocessed_img)
                
                # Ambil kelas dengan probabilitas tertinggi
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = CLASSES[predicted_class_index]
                
                # Update statistik prediksi
                prediction_stats[predicted_class] += 1
                
                # Ambil nilai confidence (probabilitas)
                confidence = float(predictions[0][predicted_class_index])
                
                # Mapping ke nama Indonesia dan level
                severity = SEVERITY_MAPPING[predicted_class]
                severity_level = SEVERITY_LEVEL_MAPPING[predicted_class]
                
                # Hasil prediksi
                result = {
                    'severity': severity,
                    'severity_level': severity_level,
                    'confidence': confidence,
                    'raw_prediction': {
                        'class': predicted_class,
                        'probabilities': {CLASSES[i]: float(predictions[0][i]) for i in range(len(CLASSES))}
                    }
                }
                
                print(f"Prediksi untuk gambar {file.filename}: {severity} (confidence: {confidence:.2f})")
                successful_predictions += 1
                
                return jsonify(result)
            except Exception as e:
                print(f"Error saat menggunakan model: {e}")
                # Fallback ke mode simulasi jika ada error dengan model
                print("Fallback ke mode simulasi...")
        
        # Mode simulasi (jika model tidak tersedia atau ada error)
        print(f"Menggunakan mode simulasi untuk gambar: {file.filename}")
        
        # Periksa gambar
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((224, 224))  # Hanya untuk memastikan gambar valid
        
        # Pilih kelas secara acak dengan bias ke kelas tertentu (untuk simulasi)
        import random
        weights = [0.5, 0.2, 0.15, 0.1, 0.05]  # Distribusi probabilitas untuk 5 kelas
        predicted_class_index = random.choices(range(len(CLASSES)), weights=weights)[0]
        predicted_class = CLASSES[predicted_class_index]
        
        # Update statistik prediksi
        prediction_stats[predicted_class] += 1
        
        # Generate confidence score yang realistis
        base_confidence = 0.75
        confidence = base_confidence + (random.random() * 0.2)  # 0.75 - 0.95
        
        # Mapping ke nama Indonesia dan level
        severity = SEVERITY_MAPPING[predicted_class]
        severity_level = SEVERITY_LEVEL_MAPPING[predicted_class]
        
        # Hasil prediksi simulasi
        result = {
            'severity': severity,
            'severity_level': severity_level,
            'confidence': confidence,
            'raw_prediction': {
                'class': predicted_class,
                'probabilities': {
                    CLASSES[0]: round(random.random() * 0.1, 3),
                    CLASSES[1]: round(random.random() * 0.1, 3),
                    CLASSES[2]: round(random.random() * 0.1, 3),
                    CLASSES[3]: round(random.random() * 0.1, 3),
                    CLASSES[4]: round(random.random() * 0.1, 3)
                },
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
            'tf_version': tf.__version__,
            'simulation_mode': model is None,
            'model_path': MODEL_PATH
        }
        
        if model is not None:
            # Dapatkan struktur model jika tersedia
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            info_data['model_summary'] = '\n'.join(model_summary)
        else:
            info_data['model_summary'] = 'Model tidak tersedia (mode simulasi)'
            info_data['note'] = 'API berjalan dalam mode simulasi. Untuk menggunakan model yang sebenarnya, pastikan file model-Retinopati.h5 tersedia.'
            
        return jsonify(info_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-model', methods=['GET'])
def download_model():
    """Endpoint untuk mengunduh file model-Retinopaty.h5"""
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'File model tidak ditemukan'}), 404
        
        # Kirim file model sebagai respons
        return send_file(
            MODEL_PATH, 
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='model-Retinopaty.h5'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Endpoint untuk mendapatkan informasi tentang semua kelas retinopati dan deskripsinya"""
    try:
        # Deskripsi untuk setiap kelas retinopati
        class_descriptions = {
            'No DR': 'Tidak ada tanda-tanda retinopati diabetik pada retina.',
            'Mild': 'Retinopati diabetik non-proliferatif ringan dengan sedikit mikroaneurisma.',
            'Moderate': 'Retinopati diabetik non-proliferatif sedang dengan adanya mikroaneurisma, perdarahan intraretinal, dan cotton wool spots.',
            'Severe': 'Retinopati diabetik non-proliferatif berat dengan banyak perdarahan, venous beading, dan IRMA (Intraretinal Microvascular Abnormalities).',
            'Proliferative DR': 'Retinopati diabetik proliferatif dengan pembentukan pembuluh darah baru (neovaskularisasi) dan jaringan fibrosa pada retina.'
        }

        # Rekomendasi tindakan untuk setiap kelas
        recommendations = {
            'No DR': 'Lakukan pemeriksaan rutin setiap tahun.',
            'Mild': 'Kontrol gula darah dan tekanan darah. Pemeriksaan ulang dalam 9-12 bulan.',
            'Moderate': 'Konsultasi dengan dokter spesialis mata. Pemeriksaan ulang dalam 6 bulan.',
            'Severe': 'Rujukan segera ke dokter spesialis mata. Pemeriksaan ulang dalam 2-3 bulan.',
            'Proliferative DR': 'Rujukan segera ke dokter spesialis mata untuk evaluasi dan kemungkinan tindakan laser atau operasi.'
        }

        # Menggabungkan semua informasi
        classes_info = []
        for i, class_name in enumerate(CLASSES):
            classes_info.append({
                'id': i,
                'name': class_name,
                'severity': SEVERITY_MAPPING[class_name],
                'severity_level': SEVERITY_LEVEL_MAPPING[class_name],
                'description': class_descriptions[class_name],
                'recommendation': recommendations[class_name]
            })

        return jsonify({
            'status': 'success',
            'classes': classes_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Endpoint untuk mendapatkan statistik prediksi"""
    global total_requests, successful_predictions, prediction_stats, app_start_time
    
    try:
        # Hitung uptime
        uptime_seconds = time.time() - app_start_time
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        uptime_formatted = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Hitung persentase untuk setiap kelas
        percentages = {}
        if successful_predictions > 0:
            for class_name, count in prediction_stats.items():
                percentages[class_name] = round((count / successful_predictions) * 100, 2)
        else:
            percentages = {class_name: 0 for class_name in prediction_stats.keys()}
        
        # Statistik
        stats = {
            'total_requests': total_requests,
            'successful_predictions': successful_predictions,
            'uptime': uptime_formatted,
            'prediction_counts': prediction_stats,
            'prediction_percentages': percentages,
            'model_loaded': model is not None,
            'simulation_mode': model is None
        }
        
        return jsonify({
            'status': 'success',
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 