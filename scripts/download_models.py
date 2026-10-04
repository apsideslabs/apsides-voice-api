#!/usr/bin/env python3
"""Download and cache all model files into MODEL_DIR.

Usage:
    python scripts/download_models.py            # everything
    python scripts/download_models.py --tts      # Kokoro only
    python scripts/download_models.py --stt      # Moonshine only
    MODEL_DIR=/data/models python scripts/download_models.py

Resulting layout (default MODEL_DIR=./models):
    models/kokoro/onnx/model_q8f16.onnx     (~86 MB)
    models/kokoro/voices/<name>.bin         (510x256 float32 each)
    models/kokoro/vocab.json                (phoneme -> id map)
    <moonshine cache>                       (managed by moonshine-voice)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

KOKORO_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_MODEL_FILE = "onnx/model_q8f16.onnx"
KOKORO_VOCAB_REPO = "hexgrad/Kokoro-82M"
KOKORO_VOCAB_FILE = "config.json"


def _model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "./models")).resolve()


def download_tts(model_dir: Path) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    kok = model_dir / "kokoro"
    kok.mkdir(parents=True, exist_ok=True)

    print(f"[tts] fetching {KOKORO_REPO}:{KOKORO_MODEL_FILE} ...")
    snapshot_download(
        repo_id=KOKORO_REPO,
        local_dir=str(kok),
        allow_patterns=[KOKORO_MODEL_FILE, "voices/*.bin"],
    )

    print(f"[tts] fetching vocab from {KOKORO_VOCAB_REPO}:{KOKORO_VOCAB_FILE} ...")
    cfg_path = hf_hub_download(
        repo_id=KOKORO_VOCAB_REPO, filename=KOKORO_VOCAB_FILE, repo_type="model"
    )
    with open(cfg_path, "r", encoding="utf-8") as fh:
        vocab = json.load(fh)["vocab"]
    with open(kok / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump({"vocab": vocab}, fh)
    print(f"[tts] done -> {kok}")


def download_stt(arch: str = "small_streaming") -> None:
    try:
        from moonshine_voice import ModelArch, get_model_for_language
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[stt] moonshine-voice is not installed: {exc}")

    names = {
        "tiny": "TINY",
        "tiny_streaming": "TINY_STREAMING",
        "base": "BASE",
        "small_streaming": "SMALL_STREAMING",
        "medium_streaming": "MEDIUM_STREAMING",
    }
    arch_enum = getattr(ModelArch, names.get(arch, "SMALL_STREAMING"))
    print(f"[stt] fetching English {arch} (q8) ...")
    model_path, model_arch = get_model_for_language(
        wanted_language="en", wanted_model_arch=arch_enum
    )
    print(f"[stt] done -> {model_path} (arch={model_arch})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Apsides Voice API models.")
    ap.add_argument("--tts", action="store_true", help="download Kokoro TTS only")
    ap.add_argument("--stt", action="store_true", help="download Moonshine STT only")
    ap.add_argument("--stt-arch", default=os.environ.get("STT_MODEL_ARCH", "small_streaming"))
    args = ap.parse_args()

    do_all = not (args.tts or args.stt)
    model_dir = _model_dir()
    print(f"MODEL_DIR = {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)

    if do_all or args.tts:
        download_tts(model_dir)
    if do_all or args.stt:
        download_stt(args.stt_arch)

    print("All requested models are ready.")


if __name__ == "__main__":
    main()
