# RetinaScan Flask API

API ini berfungsi sebagai backend untuk menganalisis gambar retina menggunakan model machine learning (H5) untuk deteksi retinopati diabetik.

## Setup

1. Pastikan Python 3.9+ sudah terpasang
2. Instal dependensi:
   ```
   pip install -r requirements.txt
   ```
   
   Jika ada masalah dengan TensorFlow, instalasi langsung melalui pip:
   ```
   pip install tensorflow==2.19.0
   ```
   
3. Pastikan file model (`model.h5`) berada di folder `backend/models/`

## Menjalankan API

```
python app.py
```

API akan berjalan pada `http://localhost:5001`

## Mode Operasi

API ini dapat berjalan dalam dua mode:
1. **Mode Model**: Menggunakan model H5 untuk prediksi yang akurat (default)
2. **Mode Simulasi**: Digunakan sebagai fallback jika model tidak tersedia atau terjadi error

## Endpoint

- `POST /predict` - Upload dan analisis gambar retina
- `GET /info` - Dapatkan informasi tentang model

## Integrasi dengan Node.js

API ini digunakan oleh aplikasi Node.js untuk menganalisis gambar retina yang diunggah pengguna. Aplikasi Node.js akan mengirim gambar ke endpoint `/predict` dan menerima hasil analisis untuk disimpan ke MongoDB.

## Catatan Versi TensorFlow

API ini menggunakan TensorFlow 2.19.0 yang merupakan versi terbaru yang tersedia. Pastikan Anda menggunakan versi Python yang kompatibel (3.9-3.12). 