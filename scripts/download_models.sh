#!/bin/bash
set -e
MODELS_DIR="$(dirname "$0")/../data/models"
mkdir -p "$MODELS_DIR"
echo "=== Downloading models for Voice Assistant ==="

# 1. GigaAM v3 CTC ONNX (ASR)
if [ ! -f "$MODELS_DIR/gigaam-v3-ctc-onnx/v3_ctc.int8.onnx" ]; then
    echo "[1/3] Downloading GigaAM v3 CTC ONNX (~220 MB)..."
    mkdir -p "$MODELS_DIR/gigaam-v3-ctc-onnx"
    cd "$MODELS_DIR/gigaam-v3-ctc-onnx"
    wget -c --tries=10 --timeout=30 "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_ctc.int8.onnx"
    wget -c --tries=10 --timeout=30 "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_vocab.txt"
    wget -c --tries=10 --timeout=30 "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/config.json"
    echo "  Done."
else
    echo "[1/3] GigaAM model already exists, skipping."
fi

# 2. Piper TTS Russian voice
if [ ! -f "$MODELS_DIR/piper-ru_RU-irina-medium/ru_RU-irina-medium.onnx" ]; then
    echo "[2/3] Downloading Piper TTS Russian voice (~50 MB)..."
    mkdir -p "$MODELS_DIR/piper-ru_RU-irina-medium"
    cd "$MODELS_DIR/piper-ru_RU-irina-medium"
    wget -c --tries=10 --timeout=30 "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"
    wget -c --tries=10 --timeout=30 "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json"
    echo "  Done."
else
    echo "[2/3] Piper TTS model already exists, skipping."
fi

# 3. Multilingual-e5-base ONNX (embeddings)
if [ ! -f "$MODELS_DIR/multilingual-e5-base-onnx/model.onnx" ]; then
    echo "[3/3] Downloading multilingual-e5-base ONNX..."
    echo "  Note: requires 'optimum' library to export."
    echo "  Run: pip install optimum[onnxruntime]"
    echo "  Then: python3 -c \""
    echo "    from optimum.onnxruntime import ORTModelForFeatureExtraction"
    echo "    from transformers import AutoTokenizer"
    echo "    model = ORTModelForFeatureExtraction.from_pretrained('intfloat/multilingual-e5-base', export=True)"
    echo "    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')"
    echo "    model.save_pretrained('$MODELS_DIR/multilingual-e5-base-onnx')"
    echo "    tokenizer.save_pretrained('$MODELS_DIR/multilingual-e5-base-onnx')"
    echo "  \""
else
    echo "[3/3] e5-base model already exists, skipping."
fi

echo ""
echo "=== All models ready ==="
