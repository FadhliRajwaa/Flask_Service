# API Retinopati Diabetik

API Flask untuk klasifikasi Retinopati Diabetik menggunakan model deep learning TensorFlow 2.19.0.

## Struktur Folder

```
backend/flask_service/
  ├── app.py                 # File utama aplikasi Flask
  ├── model-Retinopaty.h5    # Model deep learning untuk klasifikasi
  ├── requirements.txt       # Dependensi Python
  ├── Procfile              # Konfigurasi untuk deployment
  ├── render.yaml           # Konfigurasi untuk Render
  └── README.md             # Dokumentasi
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
- **Body**:
```json
{
  "image": "BASE64_ENCODED_IMAGE"
}
```
- **Response**:
```json
{
  "status": "success",
  "prediction": {
    "class": "Nama Kelas",
    "class_id": 0,
    "confidence": 0.95
  }
}
```

## Deployment ke Render

1. Buat akun di [Render](https://render.com)
2. Buat Web Service baru dan pilih "Build and deploy from a Git repository"
3. Hubungkan dengan repository GitHub Anda
4. Konfigurasi:
   - **Name**: retinopathy-api
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --timeout 180 --workers 1`
   - **Plan**: Free (atau sesuai kebutuhan)
   - **Advanced**:
     - Add Environment Variable: `TF_CPP_MIN_LOG_LEVEL` = `2`
     - Add Environment Variable: `PYTHON_VERSION` = `3.9.16`
5. Klik "Create Web Service"

## Catatan Penting

- Model membutuhkan gambar fundus mata yang diproses dengan ukuran 224x224 pixel
- Gambar harus dikirim dalam format base64
- Kelas output: ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
- API menggunakan TensorFlow 2.19.0 dan NumPy 1.26.0+ 