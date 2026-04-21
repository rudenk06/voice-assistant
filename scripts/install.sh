#!/bin/bash
set -e

echo "=== Voice Assistant Installation ==="

# 1. System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3-pip python3-venv \
    portaudio19-dev \
    libsndfile1 \
    cmake build-essential \
    wget unzip

# 2. Python environment
echo "[2/5] Setting up Python environment..."
INSTALL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$INSTALL_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download models
echo "[3/5] Downloading models..."
bash scripts/download_models.sh

# 4. Index sample documents
echo "[4/5] Indexing sample documents..."
python3 -m src.rag.indexer

# 5. Install systemd service
echo "[5/5] Installing systemd service..."
INSTALL_DIR_ESCAPED=$(echo "$INSTALL_DIR" | sed 's/\//\\\//g')
sed "s|/home/pi/voice-assistant|$INSTALL_DIR|g" systemd/voice-assistant.service | \
    sed "s|User=pi|User=$(whoami)|g" | \
    sudo tee /etc/systemd/system/voice-assistant.service > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable voice-assistant

echo ""
echo "=== Installation complete ==="
echo "Start with: sudo systemctl start voice-assistant"
echo "Or manually: source .venv/bin/activate && python -m src.main"
