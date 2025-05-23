# RetinaScan - Deteksi Retinopati Diabetik

Aplikasi ini menggunakan model deep learning untuk mendeteksi tingkat keparahan retinopati diabetik dari gambar retina.

## Cara Menjalankan Aplikasi

### Menjalankan Secara Lokal

#### Metode 1: Menggunakan streamlit_app.py (Direkomendasikan untuk Streamlit Cloud)

1. Pastikan Python terinstal
2. Instal dependensi:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```
   streamlit run streamlit_app.py
   ```
   
   Ini akan menjalankan versi aplikasi yang lebih sederhana dalam mode simulasi tanpa memerlukan TensorFlow.

#### Metode 2: Menggunakan run_local.py

1. Pastikan Python 3.7-3.9 terinstal (Python 3.9 direkomendasikan)
2. Instal dependensi:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan script run_local.py:
   ```
   python run_local.py
   ```
   Script ini akan:
   - Memeriksa semua dependensi yang diperlukan
   - Mengunduh model jika belum tersedia
   - Membuka browser secara otomatis
   - Menjalankan aplikasi Streamlit

#### Metode 3: Menjalankan Streamlit secara langsung

1. Pastikan Python 3.7-3.9 terinstal
2. Instal dependensi:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```
   streamlit run app.py
   ```
4. Buka browser dan akses `http://localhost:8501`

### Deployment di Streamlit Cloud

1. Buat akun di [Streamlit Cloud](https://streamlit.io/cloud)
2. Buat repository GitHub baru dan unggah kode aplikasi ini
3. Di Streamlit Cloud, pilih "New app" dan pilih repository GitHub Anda
4. Pilih file `streamlit_app.py` sebagai entrypoint (bukan app.py)
5. Klik "Deploy"

## Struktur File

- `app.py` - Aplikasi Streamlit utama (dengan TensorFlow)
- `streamlit_app.py` - Versi sederhana aplikasi Streamlit (mode simulasi)
- `requirements.txt` - Daftar dependensi
- `model-Retinopaty.h5` - Model terlatih untuk deteksi retinopati diabetik
- `.streamlit/` - Folder konfigurasi Streamlit
- `download_model.py` - Script untuk mengunduh model jika belum tersedia
- `run_local.py` - Script untuk menjalankan aplikasi secara lokal

## Catatan Penting

- Untuk deployment di Streamlit Cloud, gunakan file `streamlit_app.py` yang berjalan dalam mode simulasi
- Model membutuhkan gambar retina dengan ukuran 224x224 piksel
- Jika model tidak dapat dimuat, aplikasi akan berjalan dalam mode simulasi
- Untuk penggunaan produksi, pastikan model terlatih dengan baik dan divalidasi oleh ahli medis

## Troubleshooting

### Masalah TensorFlow

Jika mengalami masalah dengan instalasi TensorFlow:

1. Gunakan versi aplikasi yang lebih sederhana:
   ```
   streamlit run streamlit_app.py
   ```

2. Atau jalankan aplikasi dalam mode simulasi dengan mengatur variabel lingkungan:
   ```
   SIMULATION_MODE=1 streamlit run app.py
   ```

### Masalah Model

Jika model tidak dapat dimuat:

1. Jalankan `python download_model.py` untuk mengunduh model
2. Pastikan model berada di lokasi yang benar (direktori yang sama dengan app.py)
3. Periksa apakah model memiliki ukuran yang valid (sekitar 93MB)