# API Retinopati Diabetik

API Flask untuk klasifikasi Retinopati Diabetik menggunakan model deep learning.

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
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (atau sesuai kebutuhan)
   - **Advanced**:
     - Add Environment Variable: `TF_CPP_MIN_LOG_LEVEL` = `2`
     - Add Environment Variable: `PYTHON_VERSION` = `3.9.16`
5. Klik "Create Web Service"

## Integrasi dengan Node.js Backend

Untuk mengintegrasikan API Flask ini dengan backend Node.js, Anda dapat menggunakan fetch atau axios untuk memanggil endpoint API:

```javascript
// Contoh menggunakan axios di Node.js
const axios = require('axios');
const fs = require('fs');

// Fungsi untuk mengkonversi gambar ke base64
function imageToBase64(imagePath) {
  const image = fs.readFileSync(imagePath);
  return Buffer.from(image).toString('base64');
}

// Endpoint Flask
const FLASK_API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

// Contoh fungsi untuk prediksi retinopati
async function predictRetinopathy(imagePath) {
  try {
    const imageBase64 = imageToBase64(imagePath);
    
    const response = await axios.post(`${FLASK_API_URL}/predict`, {
      image: imageBase64
    });
    
    return response.data;
  } catch (error) {
    console.error('Error saat melakukan prediksi:', error.message);
    throw error;
  }
}

// Contoh penggunaan dalam Express route
app.post('/api/predict-retinopathy', async (req, res) => {
  try {
    const imagePath = req.file.path; // Jika menggunakan multer untuk upload
    const result = await predictRetinopathy(imagePath);
    res.json(result);
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});
```

## Catatan Penting

- Model membutuhkan gambar fundus mata yang diproses dengan ukuran 224x224 pixel
- Gambar harus dikirim dalam format base64
- Kelas output: ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
- API menggunakan CORS sehingga bisa diakses dari domain yang berbeda
- Pastikan untuk mengonfigurasi FLASK_API_URL di Node.js ke URL deployment Render Anda 