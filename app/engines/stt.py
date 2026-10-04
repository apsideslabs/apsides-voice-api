"""Moonshine Streaming Small speech-to-text engine (English, q8).

Uses the `moonshine-voice` Python package. The model is fetched on first use
into the package's local cache (or point STT_MODEL_DIR at a pre-downloaded
directory). Everything runs on CPU, with the low-latency streaming-small model
(123M params, MIT).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.errors import ModelNotReadyError

logger = logging.getLogger("apsides.voice.stt")

_ARCH_NAMES = {
    "tiny": "TINY",
    "tiny_streaming": "TINY_STREAMING",
    "base": "BASE",
    "small_streaming": "SMALL_STREAMING",
    "medium_streaming": "MEDIUM_STREAMING",
}


class MoonshineSTT:
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
        logger.info("Moonshine STT loaded (lang=%s, arch=%s)", self.language, self.arch)

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
        # moonshine-voice accepts any sample rate and resamples internally.
        transcript = self._transcriber.transcribe_without_streaming(
            audio.tolist(), sample_rate=int(sample_rate)
        )
        return self._extract_text(transcript)
