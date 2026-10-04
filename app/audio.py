"""Audio decoding/encoding helpers.

Decoding strategy (in order):
  1. `soundfile` — handles WAV/FLAC/OGG/AIFF natively (no extra binaries).
  2. `ffmpeg`    — used if present on PATH, for MP3/WebM/M4A/Opus uploads
                   (browsers' MediaRecorder often produces WebM/Opus).

Encoding: WAV via `soundfile`; MP3 via `ffmpeg` when available.
"""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile

import numpy as np

from app.errors import AudioDecodeError, UnsupportedFormatError

logger = logging.getLogger("apsides.voice.audio")


def _to_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Return (mono float32 samples in [-1, 1], sample_rate)."""
    if not data:
        raise AudioDecodeError("Empty audio payload.")

    # 1) soundfile (pure wheel, no external binaries)
    try:
        import soundfile as sf

        with sf.SoundFile(io.BytesIO(data)) as f:
            sr = int(f.samplerate)
            arr = f.read(dtype="float32", always_2d=False)
        if arr is not None and len(arr):
            return _to_mono(arr), sr
    except Exception as exc:  # noqa: BLE001 - fall through to ffmpeg
        logger.debug("soundfile could not decode upload: %s", exc)

    # 2) ffmpeg fallback for compressed containers
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                proc = subprocess.run(
                    [
                        ffmpeg, "-hide_banner", "-loglevel", "error",
                        "-i", "pipe:0", "-f", "wav", "-ac", "1", "-ar", "16000",
                        "-y", tmp.name,
                    ],
                    input=data,
                    capture_output=True,
                    timeout=60,
                )
                if proc.returncode == 0:
                    import soundfile as sf

                    arr, sr = sf.read(tmp.name, dtype="float32", always_2d=False)
                    if arr is not None and len(arr):
                        return _to_mono(arr), int(sr)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ffmpeg decode failed: %s", exc)

    raise AudioDecodeError(
        "Could not decode audio. Send WAV/FLAC/OGG, or install ffmpeg for MP3/WebM."
    )


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, _to_mono(samples), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def encode_mp3(samples: np.ndarray, sample_rate: int) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise UnsupportedFormatError(
            "MP3 output requires ffmpeg on the host. Request format='wav' instead."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav") as src, tempfile.NamedTemporaryFile(
        suffix=".mp3"
    ) as dst:
        import soundfile as sf

        sf.write(src.name, _to_mono(samples), sample_rate, format="WAV", subtype="PCM_16")
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", src.name,
             "-codec:a", "libmp3lame", "-qscale:a", "4", "-y", dst.name],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise UnsupportedFormatError("ffmpeg failed to encode MP3.")
        with open(dst.name, "rb") as fh:
            return fh.read()


def encode_audio(samples: np.ndarray, sample_rate: int, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "wav").lower()
    if fmt == "mp3":
        return encode_mp3(samples, sample_rate), "audio/mpeg"
    return encode_wav(samples, sample_rate), "audio/wav"
