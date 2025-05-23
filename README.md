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
   
3. Pastikan file model (`model-Retinopaty.h5`) berada di folder root atau di salah satu lokasi yang didukung

## Menjalankan API

```
python app.py
```

API akan berjalan pada `http://localhost:5000`

## Mode Operasi

API ini dapat berjalan dalam tiga mode:
1. **Mode Model Utama**: Menggunakan model H5 untuk prediksi yang akurat (default)
2. **Mode Model Fallback**: Menggunakan model yang lebih kecil dan efisien jika model utama gagal dimuat
3. **Mode Simulasi**: Digunakan sebagai fallback terakhir jika semua model tidak tersedia

## Pemuatan Model

API akan mencoba memuat model dengan beberapa metode:
1. Memuat model langsung dengan `load_model()`
2. Memuat arsitektur dan bobot secara terpisah
3. Memuat dari format SavedModel jika tersedia

Jika semua metode gagal, API akan membuat model fallback yang lebih kecil dan efisien.

## Endpoint

- `POST /predict` - Upload dan analisis gambar retina
- `GET /info` - Dapatkan informasi tentang model
- `GET /` - Health check
- `GET /test` - Endpoint khusus untuk testing koneksi

## Distribusi Hasil Simulasi

Jika model tidak dapat dimuat, API akan berjalan dalam mode simulasi yang memberikan hasil dengan distribusi sebagai berikut:
- 25% Tidak ada DR
- 25% Ringan
- 20% Sedang
- 20% Berat
- 10% Sangat Berat

## Integrasi dengan Node.js

API ini digunakan oleh aplikasi Node.js untuk menganalisis gambar retina yang diunggah pengguna. Aplikasi Node.js akan mengirim gambar ke endpoint `/predict` dan menerima hasil analisis untuk disimpan ke MongoDB.

## Deployment

API ini di-deploy di Render sebagai web service. Konfigurasi ada di `render.yaml`. Untuk lingkungan dengan memori terbatas seperti Render free tier, API akan menggunakan model yang lebih kecil dan efisien.

## Catatan Versi TensorFlow

API ini menggunakan TensorFlow 2.19.0 yang merupakan versi terbaru yang tersedia. Pastikan Anda menggunakan versi Python yang kompatibel (3.9-3.12).

## Perubahan Terbaru

- Perbaikan kompatibilitas TensorFlow untuk memuat model H5
- Penambahan metode alternatif untuk memuat model
- Distribusi hasil simulasi yang lebih merata (tidak lagi selalu "Sedang")
- Dukungan untuk lingkungan dengan memori terbatas
- Optimasi model fallback
- Penyimpanan model yang dioptimalkan untuk penggunaan di masa depan