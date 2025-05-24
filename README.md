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
  "confidence": 0.95
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

1. Pastikan file `render.yaml` sudah ada dan benar
2. Buat akun di [Render](https://render.com)
3. Hubungkan dengan repository GitHub Anda
4. Pilih "Blueprint" saat membuat service baru
5. Render akan otomatis menggunakan konfigurasi dari `render.yaml`
6. Tambahkan variabel lingkungan:
   - `MONGO_URI`: URI koneksi MongoDB Anda
7. Klik "Apply" untuk memulai deployment

### Catatan Penting untuk Deployment

- Pastikan model `model-Retinopaty.h5` ada dalam repository
- Model berukuran besar, jadi pastikan disk storage di Render cukup (minimal 5GB)
- Render akan menggunakan Python 3.9.16 sesuai konfigurasi
- Gunakan MongoDB Atlas untuk database produksi

## Catatan Teknis

- Model membutuhkan gambar fundus mata dengan ukuran 224x224 pixel
- Kelas output: ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
- API menggunakan TensorFlow dan Flask
- Semua prediksi disimpan dalam database MongoDB
- Tidak ada data dummy dalam aplikasi ini 