from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
import sys
from flask_cors import CORS

# Konfigurasi logging untuk TensorFlow
tf.get_logger().setLevel('INFO')
print(f"TensorFlow version: {tf.__version__}")

app = Flask(__name__)
CORS(app)

# Konfigurasi path model - gunakan path absolut untuk memastikan model ditemukan
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'model-Retinopati.h5')

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

# Kelas output model (disesuaikan dengan model yang memiliki 2 kelas)
CLASSES = ['Normal', 'Diabetic Retinopathy']
# Mapping output ke bahasa Indonesia
SEVERITY_MAPPING = {
    'Normal': 'Tidak ada',
    'Diabetic Retinopathy': 'Ada'
}
# Mapping tingkat keparahan
SEVERITY_LEVEL_MAPPING = {
    'Normal': 0,
    'Diabetic Retinopathy': 1
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        weights = [0.6, 0.4]  # Lebih sering menghasilkan kelas normal
        predicted_class_index = random.choices(range(len(CLASSES)), weights=weights)[0]
        predicted_class = CLASSES[predicted_class_index]
        
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
                    CLASSES[1]: round(random.random() * 0.1, 3)
                },
                'is_simulation': True
            }
        }
        
        # Set probabilitas kelas yang diprediksi lebih tinggi
        result['raw_prediction']['probabilities'][predicted_class] = confidence
        
        print(f"Simulasi prediksi untuk gambar {file.filename}: {severity} (confidence: {confidence:.2f})")
        
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 