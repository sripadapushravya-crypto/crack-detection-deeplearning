#!/usr/bin/env bash
# start.sh — placed in project root, used as Render start command:
#   bash start.sh
#
# Downloads best_model.pt from GitHub Releases if not already on disk,
# then starts uvicorn. The model only downloads once per Render instance
# (persists on the mounted disk between deploys if a disk is configured).

@'
#!/usr/bin/env bash
set -e

MODEL_DIR="${CNN_MODEL_DIR:-models/cnn}"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/best_model.pt" ]; then
  echo "==> Downloading best_model.pt..."
  curl -L -o "$MODEL_DIR/best_model.pt" "${MODEL_DOWNLOAD_URL}"
  echo "==> Downloaded best_model.pt ($(du -sh $MODEL_DIR/best_model.pt | cut -f1))"
fi

if [ ! -f "$MODEL_DIR/model_meta.json" ]; then
  echo "==> Downloading model_meta.json..."
  curl -L -o "$MODEL_DIR/model_meta.json" "${META_DOWNLOAD_URL}"
  echo "==> Downloaded model_meta.json"
fi

echo "==> Starting uvicorn..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
'@ | Out-File -FilePath start.sh -Encoding utf8 -NoNewline