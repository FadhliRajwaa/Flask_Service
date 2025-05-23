import streamlit as st
import numpy as np
import os
import platform
import sys
import time
import psutil
from PIL import Image
import random
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="RetinaScan - Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide"
)

# Judul dan deskripsi
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

# Status aplikasi
st.sidebar.write("Status: Mode Simulasi")
st.sidebar.write("Model: Tidak dimuat (simulasi)")

# Kelas untuk prediksi retinopati
CLASSES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}

# Fungsi untuk preprocessing gambar
def preprocess_image(image, target_size=(224, 224)):
    # Resize gambar ke target_size
    image = image.resize(target_size)
    # Konversi ke array dan normalisasi
    image_array = np.array(image) / 255.0
    return image_array

# Fungsi untuk simulasi prediksi
def simulate_prediction():
    # Simulasi hasil prediksi
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
    df = pd.DataFrame({
        'Kelas': classes,
        'Probabilitas': values
    })
    
    st.bar_chart(df.set_index('Kelas'))

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar retina", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Tombol prediksi
    if st.button("Analisis Gambar"):
        with st.spinner("Menganalisis gambar..."):
            # Simulasikan waktu pemrosesan
            time.sleep(1.5)
            
            # Lakukan simulasi prediksi
            result = simulate_prediction()
            
            # Tampilkan hasil
            display_prediction(result) 