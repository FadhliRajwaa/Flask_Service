import os
import sys
import requests
from tqdm import tqdm
import streamlit as st

# URL model (ganti dengan URL model yang sebenarnya jika tersedia)
MODEL_URL = "https://storage.googleapis.com/retinascan-models/model-Retinopaty.h5"

def download_model(url=MODEL_URL, save_path="model-Retinopaty.h5"):
    """
    Download model dari URL jika tidak tersedia secara lokal
    """
    if os.path.exists(save_path):
        print(f"Model sudah tersedia di {save_path}")
        return True
    
    try:
        print(f"Mengunduh model dari {url}...")
        response = requests.get(url, stream=True)
        
        if response.status_code != 200:
            print(f"Gagal mengunduh model. Status code: {response.status_code}")
            return False
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(save_path, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KiB', unit_scale=True):
                f.write(data)
        
        print(f"Model berhasil diunduh ke {save_path}")
        return True
    
    except Exception as e:
        print(f"Error saat mengunduh model: {e}")
        return False

def download_model_ui():
    """
    Fungsi untuk UI Streamlit untuk mengunduh model
    """
    st.title("Download Model Retinopati")
    
    if os.path.exists("model-Retinopaty.h5"):
        st.success("Model sudah tersedia!")
        model_size = os.path.getsize("model-Retinopaty.h5") / (1024 * 1024)
        st.write(f"Ukuran model: {model_size:.2f} MB")
    else:
        st.warning("Model tidak tersedia secara lokal")
        
        if st.button("Unduh Model"):
            with st.spinner("Mengunduh model... Ini mungkin memerlukan waktu beberapa menit."):
                success = download_model()
                if success:
                    st.success("Model berhasil diunduh!")
                    model_size = os.path.getsize("model-Retinopaty.h5") / (1024 * 1024)
                    st.write(f"Ukuran model: {model_size:.2f} MB")
                else:
                    st.error("Gagal mengunduh model. Silakan coba lagi nanti.")
    
    st.write("---")
    st.write("Catatan: Model ini digunakan untuk deteksi retinopati diabetik.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        # Jika dijalankan dengan Streamlit
        pass  # Streamlit akan menjalankan download_model_ui()
    else:
        # Jika dijalankan langsung dengan Python
        download_model() 