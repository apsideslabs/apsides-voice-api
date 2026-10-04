"""Model registry — loads engines once at startup and reports their state.

Loading is best-effort: if one model is missing, the API still starts and the
other endpoint keeps working. `/health` and `/ready` expose the per-model state
so a platform can decide whether to route traffic.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from app.config import Settings
from app.engines.stt import MoonshineSTT
from app.engines.tts import KokoroTTS

logger = logging.getLogger("apsides.voice.registry")


class Registry:
    def __init__(self) -> None:
        self.tts: Optional[KokoroTTS] = None
        self.stt: Optional[MoonshineSTT] = None
        self._states: Dict[str, dict] = {}
        self._started_at = time.time()

    # ------------------------------------------------------------------
    def load_all(self, settings: Settings) -> None:
        self._started_at = time.time()

        if settings.tts_enabled:
            self.tts = KokoroTTS(
                model_path=settings.resolved_tts_model_path,
                voices_dir=settings.resolved_tts_voices_dir,
                vocab_path=settings.resolved_tts_vocab_path,
                num_threads=settings.onnx_num_threads,
                default_voice=settings.tts_default_voice,
            )
            self._states["tts"] = self._safe_load("tts", self.tts.load)
        else:
            self._states["tts"] = {"enabled": False, "ready": False, "detail": "disabled"}

        if settings.stt_enabled:
            self.stt = MoonshineSTT(
                language=settings.stt_language,
                arch=settings.stt_model_arch,
                model_dir=settings.stt_model_dir,
            )
            self._states["stt"] = self._safe_load("stt", self.stt.load)
        else:
            self._states["stt"] = {"enabled": False, "ready": False, "detail": "disabled"}

    @staticmethod
    def _safe_load(name: str, loader) -> dict:
        try:
            loader()
            return {"enabled": True, "ready": True, "detail": None}
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load '%s': %s", name, exc)
            return {"enabled": True, "ready": False, "detail": str(exc)}

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, dict]:
        return dict(self._states)

    def any_ready(self) -> bool:
        return any(s.get("ready") for s in self._states.values())

    def uptime(self) -> float:
        return round(time.time() - self._started_at, 2)

    def shutdown(self) -> None:
        self.tts = None
        self.stt = None


registry = Registry()
