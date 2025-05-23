from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load trained model
model = load_model("model-Retinopaty.h5")

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
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read()))
    input_tensor = preprocess_image(image)

    predictions = model.predict(input_tensor)[0]
    class_index = predictions.argmax()
    class_name = CLASSES[class_index]

    return jsonify({
        "class": class_name,
        "confidence": float(predictions[class_index]),
        "severity_level": SEVERITY_LEVEL_MAPPING[class_name],
        "severity_description": SEVERITY_MAPPING[class_name],
        "recommendation": RECOMMENDATIONS[class_name]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
