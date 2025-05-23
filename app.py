import streamlit as st
import numpy as np
import os
import platform
import sys
import json
import time
import psutil
from PIL import Image
import io

# Coba import TensorFlow, tetapi jangan gagal jika tidak ada
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    # Print TensorFlow version untuk debugging
    st.sidebar.write(f"TensorFlow version: {tf.__version__}")
    st.sidebar.write(f"Keras version: {tf.keras.__version__}")
    
    # Set konfigurasi TensorFlow untuk menghindari error
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    TF_AVAILABLE = True
    st.sidebar.success("TensorFlow imported successfully")
except ImportError:
    TF_AVAILABLE = False
    st.sidebar.error("TensorFlow not available, running in simulation mode only")
except Exception as general_tf_error:
    st.sidebar.error(f"TensorFlow error: {general_tf_error}")
    TF_AVAILABLE = False
    st.sidebar.error("TensorFlow had error during import, running in simulation mode only")

# Cek apakah mode simulasi diaktifkan
# Jika FORCE_MODEL=1, abaikan SIMULATION_MODE dan paksa menggunakan model
# Atau jika SIMULATION_MODE tidak ditetapkan, paksa menggunakan model
force_model = os.environ.get("FORCE_MODEL") == "1"
# Periksa SIMULATION_MODE, jika tidak ada atau bukan "1", maka simulation_mode = False
simulation_mode = os.environ.get("SIMULATION_MODE") == "1" and not force_model

st.sidebar.write(f"SIMULATION_MODE env: {os.environ.get('SIMULATION_MODE')}")
st.sidebar.write(f"FORCE_MODEL env: {os.environ.get('FORCE_MODEL')}")
st.sidebar.write(f"TensorFlow available: {TF_AVAILABLE}")
st.sidebar.write(f"Simulation mode: {'ON' if simulation_mode else 'OFF'}")

# Daftar kemungkinan lokasi model
model_paths = [
    "model-Retinopaty.h5",
    "./model-Retinopaty.h5",
    "../model-Retinopaty.h5",
    "/app/model-Retinopaty.h5",
    "/app/models/model-Retinopaty.h5",
    "model/model-Retinopaty.h5",
    "./model/model-Retinopaty.h5",
    # Tambahkan lokasi model di Streamlit Cloud
    "/mount/src/retinascan/model-Retinopaty.h5",
    "/mount/src/retinascan/backend/flask_service/model-Retinopaty.h5",
    # Tambahkan lokasi model di Streamlit Share
    "/mount/src/flask_service/model-Retinopaty.h5"
]

# Fungsi untuk memuat model
@st.cache_resource
def load_model_from_path():
    # Cek semua kemungkinan lokasi
    model_path = None
    for path in model_paths:
        st.sidebar.write(f"Checking if model exists at path: {path}")
        if os.path.exists(path):
            st.sidebar.success(f"Model file found at {path}")
            model_path = path
            break

    if model_path is None:
        st.sidebar.error(f"Model file NOT found in any location")
        # Cek lokasi file saat ini
        st.sidebar.write(f"Current directory: {os.getcwd()}")
        try:
            st.sidebar.write(f"Files in current directory: {os.listdir('.')}")
            return None
        except Exception as e:
            st.sidebar.error(f"Error checking directories: {e}")
            return None

    # Coba load model jika path ditemukan
    if model_path and os.path.exists(model_path):
        # Cek ukuran file model
        model_size = os.path.getsize(model_path)
        st.sidebar.write(f"Model file size: {model_size} bytes")
        
        if model_size > 0:
            try:
                st.sidebar.info(f"Loading model from {model_path}...")
                model = load_model(model_path, compile=False)
                st.sidebar.success("Model loaded successfully")
                return model
            except Exception as load_error:
                st.sidebar.error(f"Error during model loading: {load_error}")
                return None
    return None

# Fungsi untuk preprocessing gambar
def preprocess_image(image, target_size=(224, 224)):
    # Resize gambar ke target_size
    image = image.resize(target_size)
    # Konversi ke array dan normalisasi
    image_array = np.array(image) / 255.0
    # Tambahkan dimensi batch
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Kelas untuk prediksi retinopati
CLASSES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}

# Fungsi untuk prediksi
def predict_image(model, image):
    try:
        # Preprocess gambar
        processed_image = preprocess_image(image)
        
        # Prediksi dengan model
        predictions = model.predict(processed_image)
        
        # Ambil kelas dengan probabilitas tertinggi
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Kembalikan hasil prediksi
        result = {
            "class": int(predicted_class),
            "class_name": CLASSES[predicted_class],
            "confidence": confidence,
            "probabilities": {str(i): float(p) for i, p in enumerate(predictions[0])}
        }
        return result
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return {"error": str(e)}

# Fungsi untuk simulasi prediksi (jika model tidak tersedia)
def simulate_prediction():
    # Simulasi hasil prediksi
    import random
    predicted_class = random.randint(0, 4)
    probabilities = np.zeros(5)
    probabilities[predicted_class] = random.uniform(0.7, 0.95)
    
    # Distribusikan sisa probabilitas
    remaining_prob = 1.0 - probabilities[predicted_class]
    for i in range(5):
        if i != predicted_class:
            probabilities[i] = remaining_prob / 4
    
    result = {
        "class": predicted_class,
        "class_name": CLASSES[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {str(i): float(p) for i, p in enumerate(probabilities)},
        "simulation": True
    }
    return result

# Fungsi untuk menampilkan hasil prediksi
def display_prediction(result):
    st.subheader("Hasil Diagnosis")
    
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    if "simulation" in result and result["simulation"]:
        st.warning("⚠️ Hasil ini adalah SIMULASI karena model tidak tersedia")
    
    # Tampilkan kelas prediksi dan confidence
    st.write(f"**Diagnosis:** {result['class_name']}")
    st.write(f"**Confidence:** {result['confidence']:.2%}")
    
    # Tampilkan bar chart untuk semua probabilitas
    st.subheader("Probabilitas per Kelas")
    probs = result["probabilities"]
    classes = [CLASSES[int(i)] for i in probs.keys()]
    values = list(probs.values())
    
    # Buat dataframe untuk chart
    import pandas as pd
    df = pd.DataFrame({
        'Kelas': classes,
        'Probabilitas': values
    })
    
    st.bar_chart(df.set_index('Kelas'))

# Fungsi utama Streamlit
def main():
    st.set_page_config(
        page_title="RetinaScan - Diabetic Retinopathy Detection",
        page_icon="👁️",
        layout="wide"
    )
    
    st.title("RetinaScan - Deteksi Retinopati Diabetik")
    st.write("Unggah gambar retina untuk mendeteksi tingkat retinopati diabetik")
    
    # Sidebar informasi
    st.sidebar.title("Informasi Sistem")
    st.sidebar.write(f"Platform: {platform.system()} {platform.release()}")
    st.sidebar.write(f"Python: {sys.version.split()[0]}")
    
    # Memori dan CPU info
    memory = psutil.virtual_memory()
    st.sidebar.write(f"Memory: {memory.percent}% used")
    st.sidebar.write(f"CPU: {psutil.cpu_percent()}% used")
    
    # Load model
    model = None
    if TF_AVAILABLE and not simulation_mode:
        model = load_model_from_path()
        if model is not None:
            st.sidebar.success("Model loaded successfully and ready for predictions")
        else:
            st.sidebar.warning("Failed to load model, will use simulation mode")
            simulation_mode = True
    
    # Upload gambar
    uploaded_file = st.file_uploader("Pilih gambar retina", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Tampilkan gambar
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Tombol prediksi
        if st.button("Analisis Gambar"):
            with st.spinner("Menganalisis gambar..."):
                # Lakukan prediksi atau simulasi
                if model is not None and not simulation_mode:
                    result = predict_image(model, image)
                else:
                    st.warning("Model tidak tersedia, menggunakan mode simulasi")
                    result = simulate_prediction()
                
                # Tampilkan hasil
                display_prediction(result)

if __name__ == "__main__":
    main()
