#!/usr/bin/env bash
# start.sh — placed in project root, used as Render start command:
#   bash start.sh
#
# Downloads best_model.pt from GitHub Releases if not already on disk,
# then starts uvicorn. The model only downloads once per Render instance
# (persists on the mounted disk between deploys if a disk is configured).

set -e

MODEL_DIR="${CNN_MODEL_DIR:-models/cnn}"
MODEL_PATH="$MODEL_DIR/best_model.pt"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
  echo "==> Model not found at $MODEL_PATH — downloading..."
  # Replace this URL with your actual GitHub Release asset URL (see instructions)
  MODEL_URL="${MODEL_DOWNLOAD_URL:-}"

  if [ -z "$MODEL_URL" ]; then
    echo "ERROR: MODEL_DOWNLOAD_URL env var not set. Cannot download model."
    echo "Set it in Render environment variables to the direct download URL of best_model.pt"
    exit 1
  fi

  curl -L -o "$MODEL_PATH" "$MODEL_URL"
  echo "==> Model downloaded successfully ($(du -sh $MODEL_PATH | cut -f1))"
else
  echo "==> Model already present at $MODEL_PATH — skipping download"
fi

echo "==> Starting uvicorn..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
