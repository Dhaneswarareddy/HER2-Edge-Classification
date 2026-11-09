#!/bin/bash
# Run once on the Pi from the extracted bundle. Needs 64-bit Raspberry Pi OS (Bookworm).
set -e
[ "$(uname -m)" = "aarch64" ] || { echo "This needs a 64-bit OS (aarch64), found $(uname -m)."; exit 1; }
sudo apt update && sudo apt install -y python3-venv
python3 -m venv "$HOME/her2env"
source "$HOME/her2env/bin/activate"
pip install --upgrade pip
pip install -r requirements_pi.txt || { echo "Trying tflite-runtime instead of ai-edge-litert"; pip install "numpy<2.0" "pillow>=10" psutil tflite-runtime; }
python3 verify_bundle.py
