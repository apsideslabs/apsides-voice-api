"""Model download / caching logic.

Shared by `scripts/download_models.py` (CLI) and the app startup hook
(`app.registry`). Nothing here writes into git — everything lands under
MODEL_DIR, which is git-ignored and expected to be a persistent volume.
"""
from __future__ import annotations

import json
import logging
import platform
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger("apsides.voice.models")

# --- Kokoro TTS ------------------------------------------------------------
KOKORO_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_MODEL_FILE = "onnx/model_q8f16.onnx"
KOKORO_VOCAB_REPO = "hexgrad/Kokoro-82M"
KOKORO_VOCAB_FILE = "config.json"

# --- Moonshine STT via transcribe.cpp -------------------------------------
TRANSCRIBE_GH = "https://github.com/handy-computer/transcribe.cpp/releases/download"
_TRANSCRIBE_PLATFORM = {"x86_64": "linux-x86_64", "aarch64": "linux-aarch64"}
GGUF_REPO = "handy-computer/moonshine-streaming-small-gguf"
GGUF_FILE = "moonshine-streaming-small-Q8_0.gguf"

# --- Moonshine STT via moonshine-voice (alternative backend) ---------------
_ARCH_NAMES = {
    "tiny": "TINY",
    "tiny_streaming": "TINY_STREAMING",
    "base": "BASE",
    "small_streaming": "SMALL_STREAMING",
    "medium_streaming": "MEDIUM_STREAMING",
}


def _download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            logger.info("downloading %s (attempt %d/%d)", url, attempt, retries)
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            tmp.replace(dest)
            logger.info("saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("download failed (%s)", exc)
            if attempt == retries:
                raise
            tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------
def download_kokoro(model_dir: Path) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    kok = model_dir / "kokoro"
    kok.mkdir(parents=True, exist_ok=True)

    logger.info("[tts] fetching %s", KOKORO_MODEL_FILE)
    snapshot_download(
        repo_id=KOKORO_REPO,
        local_dir=str(kok),
        allow_patterns=[KOKORO_MODEL_FILE, "voices/*.bin"],
    )

    logger.info("[tts] fetching vocab")
    cfg_path = hf_hub_download(
        repo_id=KOKORO_VOCAB_REPO, filename=KOKORO_VOCAB_FILE, repo_type="model"
    )
    with open(cfg_path, "r", encoding="utf-8") as fh:
        vocab = json.load(fh)["vocab"]
    with open(kok / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump({"vocab": vocab}, fh)
    logger.info("[tts] ready -> %s", kok)


def download_transcribe_native(model_dir: Path, version: str) -> None:
    machine = platform.machine()
    plat = _TRANSCRIBE_PLATFORM.get(machine)
    if plat is None:
        raise RuntimeError(f"no prebuilt transcribe.cpp bundle for arch '{machine}'")

    tdir = model_dir / "transcribe"
    lib = tdir / f"transcribe-native-{plat}-cpu-vulkan" / "libtranscribe.so"
    if lib.exists():
        logger.info("[stt] transcribe native already present -> %s", lib)
        return

    name = f"transcribe-native-{version}-{plat}-cpu-vulkan.tar.gz"
    url = f"{TRANSCRIBE_GH}/v{version}/{name}"
    tgz = model_dir / name
    _download(url, tgz)

    logger.info("[stt] extracting %s", tgz.name)
    tdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz) as tar:
        tar.extractall(tdir)
    tgz.unlink(missing_ok=True)
    logger.info("[stt] transcribe native ready -> %s", tdir)


def download_gguf(model_dir: Path, filename: str = GGUF_FILE) -> Path:
    from huggingface_hub import hf_hub_download

    dst = model_dir / "moonshine" / filename
    if dst.exists():
        logger.info("[stt] gguf already present -> %s", dst)
        return dst
    logger.info("[stt] fetching %s:%s", GGUF_REPO, filename)
    path = hf_hub_download(repo_id=GGUF_REPO, filename=filename, local_dir=str(dst.parent))
    return Path(path)


def download_moonshine_voice(arch: str = "small_streaming") -> None:
    """Alternative STT backend (pip `moonshine-voice`)."""
    try:
        from moonshine_voice import ModelArch, get_model_for_language
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"moonshine-voice is not installed: {exc}") from exc
    arch_enum = getattr(ModelArch, _ARCH_NAMES.get(arch, "SMALL_STREAMING"))
    model_path, model_arch = get_model_for_language(
        wanted_language="en", wanted_model_arch=arch_enum
    )
    logger.info("[stt] moonshine-voice ready -> %s (arch=%s)", model_path, model_arch)


# --------------------------------------------------------------------------
def _transcribe_native_via_pip() -> bool:
    """True when `transcribe-cpp-native` is installed (it bundles libtranscribe.so).

    `pip install transcribe-cpp` pulls this in automatically, so the separate
    GitHub native bundle is only a fallback for hosts where pip can't fetch the
    platform wheel.
    """
    try:
        import transcribe_cpp_native  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_models(settings, which: str = "all") -> None:
    """Download whatever is missing. Never raises fatally — logs and moves on
    so the API can still boot in a degraded state."""
    model_dir = Path(settings.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if which in ("all", "tts") and settings.tts_enabled:
        if not Path(settings.resolved_tts_model_path).exists():
            try:
                download_kokoro(model_dir)
            except Exception as exc:  # noqa: BLE001
                logger.error("[tts] download failed: %s", exc)

    if which in ("all", "stt") and settings.stt_enabled:
        try:
            if settings.stt_backend == "transcribe_cpp":
                if _transcribe_native_via_pip():
                    logger.info("[stt] native runtime provided by transcribe-cpp-native (pip)")
                else:
                    logger.info("[stt] no pip native package; fetching native bundle")
                    download_transcribe_native(model_dir, settings.transcribe_version)
                download_gguf(model_dir)
            else:
                download_moonshine_voice(settings.stt_model_arch)
        except Exception as exc:  # noqa: BLE001
            logger.error("[stt] download failed: %s", exc)
