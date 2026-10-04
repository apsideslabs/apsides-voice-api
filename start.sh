#!/usr/bin/env bash
# Start script for hosts that run a shell command (Botkeep, Render, etc.).
# Downloads models on first boot (cached under MODEL_DIR), then serves the API.
set -euo pipefail

cd "$(dirname "$0")"

export MODEL_DIR="${MODEL_DIR:-./models}"
export PORT="${PORT:-8000}"
export STT_BACKEND="${STT_BACKEND:-transcribe_cpp}"

# --- Fetch models once (idempotent) ---------------------------------------
if [ ! -f "${MODEL_DIR}/kokoro/onnx/model_q8f16.onnx" ]; then
  echo "[start] downloading models into ${MODEL_DIR} ..."
  python scripts/download_models.py || echo "[start] WARNING: model download failed; starting anyway"
fi

# --- Point transcribe.cpp at a native library -----------------------------
# Normally `pip install transcribe-cpp` provides libtranscribe.so (via
# transcribe-cpp-native) and auto-discovery finds it. Only set TRANSCRIBE_LIBRARY
# explicitly if a native bundle was downloaded into MODEL_DIR/transcribe.
if [ "${STT_BACKEND}" = "transcribe_cpp" ] && [ -z "${TRANSCRIBE_LIBRARY:-}" ]; then
  LIB="$(find "${MODEL_DIR}/transcribe" -name libtranscribe.so 2>/dev/null | head -n1 || true)"
  if [ -n "${LIB}" ]; then
    export TRANSCRIBE_LIBRARY="${LIB}"
    echo "[start] TRANSCRIBE_LIBRARY=${TRANSCRIBE_LIBRARY}"
  else
    echo "[start] using pip-provided native runtime (auto-discovered)"
  fi
fi

echo "[start] serving on 0.0.0.0:${PORT}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --workers 1
