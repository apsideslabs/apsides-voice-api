#!/usr/bin/env python3
"""Download and cache all model files into MODEL_DIR.

Usage:
    python scripts/download_models.py            # everything
    python scripts/download_models.py --tts      # Kokoro only
    python scripts/download_models.py --stt      # STT only
    MODEL_DIR=/data/models python scripts/download_models.py

Resulting layout (default MODEL_DIR=./models):
    models/kokoro/onnx/model_q8f16.onnx                              (~86 MB)
    models/kokoro/voices/<name>.bin                                  (510x256 f32 each)
    models/kokoro/vocab.json                                         (phoneme -> id)
    models/transcribe/transcribe-native-linux-x86_64-cpu-vulkan/     (~61 MB, libtranscribe.so)
    models/moonshine/moonshine-streaming-small-Q8_0.gguf             (~189 MB)

Nothing is written into git — all of this lives under MODEL_DIR.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `app` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.model_download import (  # noqa: E402
    download_gguf,
    download_kokoro,
    download_moonshine_voice,
    download_transcribe_native,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Apsides Voice API models.")
    ap.add_argument("--tts", action="store_true", help="download Kokoro TTS only")
    ap.add_argument("--stt", action="store_true", help="download STT only")
    args = ap.parse_args()

    do_all = not (args.tts or args.stt)
    model_dir = Path(settings.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"MODEL_DIR = {model_dir}")
    print(f"STT_BACKEND = {settings.stt_backend}")

    if do_all or args.tts:
        download_kokoro(model_dir)

    if do_all or args.stt:
        if settings.stt_backend == "moonshine_voice":
            download_moonshine_voice(settings.stt_model_arch)
        else:
            download_transcribe_native(model_dir, settings.transcribe_version)
            download_gguf(model_dir)

    print("All requested models are ready.")


if __name__ == "__main__":
    main()
