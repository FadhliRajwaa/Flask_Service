# API Retinopati Diabetik

API Flask untuk klasifikasi Retinopati Diabetik menggunakan model deep learning TensorFlow.

## Struktur Folder

```
backend/flask_service/
  ├── app.py                 # File utama aplikasi Flask
  ├── model-Retinopaty.h5    # Model deep learning untuk klasifikasi
  ├── requirements.txt       # Dependensi Python
  ├── Procfile               # Konfigurasi untuk deployment
  ├── render.yaml            # Konfigurasi untuk Render
  └── README.md              # Dokumentasi
```

## Penggunaan Lokal

1. Install dependensi:
```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi:
```bash
python app.py
```

3. API akan berjalan di `http://localhost:5000`

## Endpoint API

### 1. Health Check
- **URL**: `/`
- **Method**: `GET`
- **Response**: Status API dan status model

### 2. Prediksi Retinopati
- **URL**: `/predict`
- **Method**: `POST`
- **Body**: Form-data dengan key 'file' dan value berupa file gambar
- **Response**:
```json
{
  "id": "prediction_id",
  "class": "Nama Kelas",
  "confidence": 0.95,
  "simulation_mode": false
}
```

### 3. Mendapatkan Daftar Prediksi
- **URL**: `/predictions?page=1&limit=20`
- **Method**: `GET`
- **Response**: Daftar prediksi dengan pagination

### 4. Statistik Prediksi
- **URL**: `/stats`
- **Method**: `GET`
- **Response**: Statistik prediksi berdasarkan kelas dan waktu

### 5. Test Model
- **URL**: `/test-model`
- **Method**: `GET`
- **Response**: Status model dan ringkasan model

## Deployment ke Render

### Langkah-langkah Deployment

1. Pastikan file `render.yaml` sudah ada dan benar
2. Buat akun di [Render](https://render.com)
3. Hubungkan dengan repository GitHub Anda
4. Pilih "Blueprint" saat membuat service baru
5. Render akan otomatis menggunakan konfigurasi dari `render.yaml`

### Konfigurasi Variabel Lingkungan

Tambahkan variabel lingkungan berikut di Render Dashboard:
- `MONGO_URI`: URI koneksi MongoDB Anda (contoh: `mongodb+srv://username:password@cluster.mongodb.net/database`)
- `FLASK_ENV`: `production`
- `FLASK_DEBUG`: `0`
- `PORT`: `10000`
- `TF_FORCE_GPU_ALLOW_GROWTH`: `true`
- `TF_CPP_MIN_LOG_LEVEL`: `0`
- `PYTHONUNBUFFERED`: `true`

### Troubleshooting Deployment

1. **Cold Start**: Render Free Tier memiliki cold start time. Tunggu 2-3 menit setelah deployment untuk memastikan service berjalan.

2. **Model Loading Error**: Jika model gagal dimuat:
   - Pastikan model `model-Retinopaty.h5` ada dalam repository
   - Cek log di Render Dashboard untuk error spesifik
   - Aplikasi akan otomatis beralih ke mode simulasi jika model gagal dimuat

3. **502 Bad Gateway**: Biasanya terjadi selama cold start. Tunggu beberapa menit dan coba lagi.

4. **Koneksi MongoDB**: Pastikan URI MongoDB valid dan dapat diakses dari Render.

5. **Disk Space**: Pastikan konfigurasi disk di `render.yaml` cukup (minimal 5GB) untuk model.

### Memverifikasi Deployment

1. Kunjungi URL health check endpoint: `https://your-service-name.onrender.com/`
2. Respons harus menampilkan status `online` dan informasi model
3. Jika `model_loaded` adalah `false`, periksa log untuk error loading model

## Integrasi dengan Backend Node.js

Untuk mengintegrasikan dengan backend Node.js:

1. Atur variabel lingkungan `FLASK_API_URL` di backend Node.js ke URL Flask service
2. Gunakan endpoint `/predict` untuk mengirim gambar untuk analisis
3. Tangani respons termasuk kemungkinan mode simulasi

## Catatan Teknis

- Model membutuhkan gambar fundus mata dengan ukuran 224x224 pixel
- Kelas output: ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
- API menggunakan TensorFlow dan Flask
- Semua prediksi disimpan dalam database MongoDB
- Mode simulasi akan aktif jika model gagal dimuat 