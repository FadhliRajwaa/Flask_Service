import os
import sys
import subprocess
import webbrowser

def check_dependencies():
    """
    Memeriksa apakah semua dependensi telah terinstal
    """
    try:
        import streamlit
        import tensorflow
        import numpy
        import pandas
        import matplotlib
        import PIL
        import psutil
        import tqdm
        import requests
        
        print("Semua dependensi telah terinstal.")
        return True
    except ImportError as e:
        print(f"Error: {e}")
        print("Beberapa dependensi belum terinstal.")
        print("Jalankan 'pip install -r requirements.txt' untuk menginstal semua dependensi.")
        return False

def check_model():
    """
    Memeriksa apakah model tersedia
    """
    if os.path.exists("model-Retinopaty.h5"):
        print("Model ditemukan.")
        return True
    else:
        print("Model tidak ditemukan.")
        print("Menjalankan download_model.py untuk mengunduh model...")
        try:
            from download_model import download_model
            success = download_model()
            if success:
                print("Model berhasil diunduh.")
                return True
            else:
                print("Gagal mengunduh model.")
                print("Aplikasi akan berjalan dalam mode simulasi.")
                return False
        except Exception as e:
            print(f"Error saat mengunduh model: {e}")
            print("Aplikasi akan berjalan dalam mode simulasi.")
            return False

def run_app():
    """
    Menjalankan aplikasi Streamlit
    """
    print("Menjalankan aplikasi Streamlit...")
    
    # Buka browser secara otomatis
    webbrowser.open("http://localhost:8501")
    
    # Jalankan Streamlit
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    print("RetinaScan - Deteksi Retinopati Diabetik")
    print("=======================================")
    
    if check_dependencies():
        check_model()  # Cek model, tapi tetap lanjutkan meskipun model tidak tersedia
        run_app()
    else:
        print("Aplikasi tidak dapat dijalankan karena dependensi belum lengkap.") 