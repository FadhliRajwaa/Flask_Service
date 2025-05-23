# RetinaScan - Deteksi Retinopati Diabetik

Aplikasi ini menggunakan model deep learning untuk mendeteksi tingkat keparahan retinopati diabetik dari gambar retina.

## Cara Menjalankan Aplikasi

### Menjalankan Secara Lokal

#### Metode 1: Menggunakan run_local.py (Direkomendasikan)

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

#### Metode 2: Menjalankan Streamlit secara langsung

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
3. Pastikan file model `model-Retinopaty.h5` tersedia di repository atau di cloud storage
4. Di Streamlit Cloud, pilih "New app" dan pilih repository GitHub Anda
5. Pilih file `app.py` sebagai entrypoint
6. Klik "Deploy"

## Struktur File

- `app.py` - Aplikasi Streamlit utama
- `requirements.txt` - Daftar dependensi
- `model-Retinopaty.h5` - Model terlatih untuk deteksi retinopati diabetik
- `.streamlit/` - Folder konfigurasi Streamlit
- `download_model.py` - Script untuk mengunduh model jika belum tersedia
- `run_local.py` - Script untuk menjalankan aplikasi secara lokal

## Catatan Penting

- Model membutuhkan gambar retina dengan ukuran 224x224 piksel
- Jika model tidak dapat dimuat, aplikasi akan berjalan dalam mode simulasi
- Untuk penggunaan produksi, pastikan model terlatih dengan baik dan divalidasi oleh ahli medis
- Aplikasi ini memerlukan TensorFlow yang kompatibel dengan versi Python Anda
- Jika mengalami masalah dengan TensorFlow, coba gunakan versi Python 3.9 yang direkomendasikan

## Troubleshooting

### Masalah TensorFlow

Jika mengalami masalah dengan instalasi TensorFlow:

1. Pastikan menggunakan Python 3.7-3.9 (direkomendasikan Python 3.9)
2. Coba instal versi TensorFlow yang sesuai:
   ```
   pip install tensorflow-cpu==2.9.1
   ```
3. Jika masih mengalami masalah, jalankan aplikasi dalam mode simulasi dengan mengatur variabel lingkungan:
   ```
   SIMULATION_MODE=1 streamlit run app.py
   ```

### Masalah Model

Jika model tidak dapat dimuat:

1. Jalankan `python download_model.py` untuk mengunduh model
2. Pastikan model berada di lokasi yang benar (direktori yang sama dengan app.py)
3. Periksa apakah model memiliki ukuran yang valid (sekitar 93MB)