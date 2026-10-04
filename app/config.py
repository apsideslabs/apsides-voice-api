"""Configuration and environment handling.

Every tunable is an environment variable (see `.env.example`). Nothing secret
is ever hardcoded here — secrets such as API_KEY are read from the environment
only and are never committed.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ------------------------------------------------------------
    app_name: str = "apsides-voice-api"
    version: str = "1.1.0"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, description="HTTP port; hosts set $PORT")
    log_level: str = "INFO"

    # --- Storage --------------------------------------------------------
    # Root directory for downloaded model files. Point this at a persistent
    # volume on the host (e.g. /data/models) so downloads survive restarts.
    model_dir: str = "./models"
    # Download missing models during app startup. Handy on hosts that only run
    # `uvicorn main:app`; set false if you download via start.sh / build step.
    auto_download_models: bool = True

    # --- TTS (Kokoro-82M ONNX q8f16) ------------------------------------
    tts_enabled: bool = True
    tts_model_path: Optional[str] = None   # default: <model_dir>/kokoro/onnx/model_q8f16.onnx
    tts_voices_dir: Optional[str] = None   # default: <model_dir>/kokoro/voices
    tts_vocab_path: Optional[str] = None   # default: <model_dir>/kokoro/vocab.json
    tts_default_voice: str = "af_heart"
    tts_max_chars: int = 1000
    tts_speed_min: float = 0.5
    tts_speed_max: float = 2.0

    # --- STT ------------------------------------------------------------
    stt_enabled: bool = True
    # Which STT runtime to use: "transcribe_cpp" (GGUF q8_0) or "moonshine_voice".
    stt_backend: str = "transcribe_cpp"
    stt_language: str = "en"
    stt_max_upload_mb: int = 25

    # transcribe.cpp backend (Moonshine Streaming Small Q8_0 GGUF)
    stt_model_path: Optional[str] = None       # default: <model_dir>/moonshine/moonshine-streaming-small-Q8_0.gguf
    transcribe_version: str = "0.3.0"          # GitHub release version for the native bundle
    transcribe_native_dir: Optional[str] = None  # default: <model_dir>/transcribe
    transcribe_library: Optional[str] = None   # explicit path to libtranscribe.so

    # moonshine-voice backend (alternative)
    stt_model_arch: str = "small_streaming"    # tiny|tiny_streaming|base|small_streaming|medium_streaming
    stt_model_dir: Optional[str] = None        # default: moonshine-voice cache

    # --- Runtime / resource controls ------------------------------------
    onnx_num_threads: int = 1        # keep at 1 on 0.5-1.5 vCPU hosts
    max_concurrency_tts: int = 1     # simultaneous TTS jobs
    max_concurrency_stt: int = 1     # simultaneous STT jobs
    request_timeout_s: int = 120

    # --- Security -------------------------------------------------------
    # If API_KEY is set, every /tts and /stt call must send it in X-API-Key.
    api_key: Optional[str] = None
    cors_origins: str = "*"          # comma-separated list, or "*"
    allow_credentials: bool = False

    # ------------------------------------------------------------------
    @property
    def cors_origin_list(self) -> List[str]:
        raw = (self.cors_origins or "*").strip()
        if raw in ("", "*"):
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    # Resolved paths ----------------------------------------------------
    @property
    def resolved_tts_model_path(self) -> str:
        return self.tts_model_path or os.path.join(
            self.model_dir, "kokoro", "onnx", "model_q8f16.onnx"
        )

    @property
    def resolved_tts_voices_dir(self) -> str:
        return self.tts_voices_dir or os.path.join(self.model_dir, "kokoro", "voices")

    @property
    def resolved_tts_vocab_path(self) -> str:
        return self.tts_vocab_path or os.path.join(self.model_dir, "kokoro", "vocab.json")

    @property
    def resolved_stt_model_path(self) -> str:
        return self.stt_model_path or os.path.join(
            self.model_dir, "moonshine", "moonshine-streaming-small-Q8_0.gguf"
        )

    @property
    def resolved_transcribe_native_dir(self) -> str:
        return self.transcribe_native_dir or os.path.join(self.model_dir, "transcribe")

    def resolve_transcribe_library(self) -> Optional[str]:
        """Find libtranscribe.so: explicit path first, then the native dir."""
        if self.transcribe_library and Path(self.transcribe_library).exists():
            return self.transcribe_library
        root = Path(self.resolved_transcribe_native_dir)
        if root.exists():
            for so in sorted(root.rglob("libtranscribe.so")):
                return str(so)
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
