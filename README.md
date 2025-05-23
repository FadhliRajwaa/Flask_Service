# RetinaScan - Deteksi Retinopati Diabetik

Aplikasi ini menggunakan model deep learning untuk mendeteksi tingkat keparahan retinopati diabetik dari gambar retina.

## Cara Menjalankan Aplikasi

### Menjalankan Secara Lokal

1. Pastikan Python 3.7+ terinstal
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

## Catatan Penting

- Model membutuhkan gambar retina dengan ukuran 224x224 piksel
- Jika model tidak dapat dimuat, aplikasi akan berjalan dalam mode simulasi
- Untuk penggunaan produksi, pastikan model terlatih dengan baik dan divalidasi oleh ahli medis