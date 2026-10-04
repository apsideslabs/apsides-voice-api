#!/usr/bin/env bash
# Start script for hosts that run a shell command (Botkeep, Render, etc.).
# Downloads models on first boot (cached under MODEL_DIR), then serves the API.
set -euo pipefail

cd "$(dirname "$0")"

export MODEL_DIR="${MODEL_DIR:-./models}"
export PORT="${PORT:-8000}"

if [ ! -f "${MODEL_DIR}/kokoro/onnx/model_q8f16.onnx" ]; then
  echo "[start] models not found in ${MODEL_DIR}; downloading..."
  python scripts/download_models.py || echo "[start] model download failed; starting with whatever is available"
fi

echo "[start] serving on 0.0.0.0:${PORT}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --workers 1
