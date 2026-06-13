#!/usr/bin/env bash
set -e

MODEL_DIR="${CNN_MODEL_DIR:-models/cnn}"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/best_model.pt" ]; then
  echo "==> Downloading best_model.pt..."
  curl -L -o "$MODEL_DIR/best_model.pt" "${MODEL_DOWNLOAD_URL}"
  echo "==> Downloaded best_model.pt"
else
  echo "==> best_model.pt already present"
fi

if [ ! -f "$MODEL_DIR/model_meta.json" ]; then
  echo "==> Downloading model_meta.json..."
  curl -L -o "$MODEL_DIR/model_meta.json" "${META_DOWNLOAD_URL}"
  echo "==> Downloaded model_meta.json"
else
  echo "==> model_meta.json already present"
fi

echo "==> Starting uvicorn..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
