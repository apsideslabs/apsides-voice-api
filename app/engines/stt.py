"""English speech-to-text engines.

Two interchangeable backends, selected by `STT_BACKEND`:

* ``transcribe_cpp`` (default) — Moonshine Streaming Small **Q8_0 GGUF** run by
  the `transcribe-cpp` Python bindings on the ggml CPU runtime. Model files:
  `moonshine-streaming-small-Q8_0.gguf` (~189 MB) + a prebuilt native bundle.
* ``moonshine_voice`` — the same upstream Moonshine Streaming Small model,
  packaged as `.ort` by the `moonshine-voice` pip package (simpler, but a
  different runtime).

Both are CPU-only, English-only, MIT.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app.errors import ModelNotReadyError

logger = logging.getLogger("apsides.voice.stt")

TARGET_SR = 16000
_ARCH_NAMES = {
    "tiny": "TINY",
    "tiny_streaming": "TINY_STREAMING",
    "base": "BASE",
    "small_streaming": "SMALL_STREAMING",
    "medium_streaming": "MEDIUM_STREAMING",
}


def _resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    """transcribe.cpp wants mono 16 kHz float32; moonshine-voice accepts any SR.

    Box filter for integer downsampling (the common 48/32/8 kHz -> 16 kHz cases)
    and linear interpolation otherwise, so there is no scipy dependency.
    """
    x = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    if sr == TARGET_SR or sr <= 0 or x.size == 0:
        return x
    if sr % TARGET_SR == 0:
        d = sr // TARGET_SR
        n = x.size - (x.size % d)
        if n:
            return x[:n].reshape(-1, d).mean(axis=1).astype(np.float32)
    n_out = max(1, int(round(x.size * TARGET_SR / sr)))
    xi = np.linspace(0.0, x.size - 1, n_out)
    return np.interp(xi, np.arange(x.size), x).astype(np.float32)


class TranscribeCppSTT:
    """Moonshine Streaming Small Q8_0 via transcribe.cpp (GGUF, CPU)."""

    name = "transcribe_cpp"

    def __init__(self, model_path: str, library_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        self.library_path = library_path
        self._tc = None
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"STT model not found: {self.model_path}")
        if self.library_path:
            os.environ["TRANSCRIBE_LIBRARY"] = self.library_path
        # Prefer CPU explicitly so the loader never tries the Vulkan backend.
        os.environ.setdefault("TRANSCRIBE_BACKEND", "cpu")

        try:
            import transcribe_cpp
        except Exception as exc:  # noqa: BLE001
            raise ModelNotReadyError(
                "transcribe-cpp is not importable. Install it (`pip install transcribe-cpp`) "
                "and ensure the native bundle is present (TRANSCRIBE_LIBRARY / native dir)."
            ) from exc

        self._tc = transcribe_cpp
        device = self._pick_cpu_device(transcribe_cpp)
        try:
            self._model = (
                transcribe_cpp.Model(str(self.model_path), device=device)
                if device is not None
                else transcribe_cpp.Model(str(self.model_path))
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelNotReadyError(f"Failed to open GGUF model: {exc}") from exc
        logger.info("transcribe.cpp STT loaded -> %s", self.model_path)

    @staticmethod
    def _pick_cpu_device(tc):
        try:
            for dev in tc.backends():
                if getattr(dev, "device_type", None) == "cpu":
                    return dev
        except Exception as exc:  # noqa: BLE001
            logger.debug("backend enumeration failed: %s", exc)
        return None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if not self.ready:
            raise ModelNotReadyError("STT model is not loaded.")
        pcm = _resample_to_16k(samples, int(sample_rate))
        with self._lock:  # 0.x: one run at a time per Model
            with self._model.session() as session:
                result = session.run(pcm)
        return (getattr(result, "text", None) or "").strip()


class MoonshineVoiceSTT:
    """Moonshine Streaming Small via the moonshine-voice pip package (.ort)."""

    name = "moonshine_voice"

    def __init__(self, language: str = "en", arch: str = "small_streaming",
                 model_dir: Optional[str] = None) -> None:
        self.language = language
        self.arch = arch
        self.model_dir = Path(model_dir) if model_dir else None
        self._transcriber = None

    def load(self) -> None:
        try:
            from moonshine_voice import ModelArch, Transcriber, get_model_for_language
        except Exception as exc:  # noqa: BLE001
            raise ModelNotReadyError(
                "moonshine-voice is not installed. Run `pip install moonshine-voice`."
            ) from exc
        arch_enum = getattr(ModelArch, _ARCH_NAMES.get(self.arch, "SMALL_STREAMING"))
        if self.model_dir and self.model_dir.exists():
            model_path, model_arch = str(self.model_dir), arch_enum
        else:
            model_path, model_arch = get_model_for_language(
                wanted_language=self.language, wanted_model_arch=arch_enum
            )
        self._transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
        logger.info("moonshine-voice STT loaded (lang=%s, arch=%s)", self.language, self.arch)

    @property
    def ready(self) -> bool:
        return self._transcriber is not None

    @staticmethod
    def _extract_text(transcript) -> str:
        text = getattr(transcript, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        lines = getattr(transcript, "lines", None) or []
        parts = []
        for line in lines:
            piece = getattr(line, "text", None) or (line if isinstance(line, str) else "")
            if piece:
                parts.append(str(piece).strip())
        return " ".join(p for p in parts if p).strip()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if not self.ready:
            raise ModelNotReadyError("STT model is not loaded.")
        audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        transcript = self._transcriber.transcribe_without_streaming(
            audio.tolist(), sample_rate=int(sample_rate)
        )
        return self._extract_text(transcript)


def build_stt(settings):
    """Construct the configured STT backend (not yet loaded)."""
    if settings.stt_backend == "moonshine_voice":
        return MoonshineVoiceSTT(
            language=settings.stt_language,
            arch=settings.stt_model_arch,
            model_dir=settings.stt_model_dir,
        )
    return TranscribeCppSTT(
        model_path=settings.resolved_stt_model_path,
        library_path=settings.resolve_transcribe_library(),
    )
