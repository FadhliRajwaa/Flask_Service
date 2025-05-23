#!/bin/bash
set -e

# Install dependensi
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt

# Khusus untuk h5py (jika diperlukan)
pip install h5py==3.1.0 --no-build-isolation

echo "Build completed successfully!" 