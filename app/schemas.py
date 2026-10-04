"""Request/response models with validation."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize.")
    voice: Optional[str] = Field(
        default=None, description="Kokoro voice id, e.g. 'af_heart'."
    )
    speed: float = Field(default=1.0, description="Speaking rate multiplier.")
    format: str = Field(default="wav", description="Audio container: 'wav' or 'mp3'.")

    @field_validator("text")
    @classmethod
    def _check_text(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > settings.tts_max_chars:
            raise ValueError(
                f"text too long ({len(v)} chars); limit is {settings.tts_max_chars}"
            )
        return v

    @field_validator("speed")
    @classmethod
    def _check_speed(cls, v: float) -> float:
        if not (settings.tts_speed_min <= v <= settings.tts_speed_max):
            raise ValueError(
                f"speed must be between {settings.tts_speed_min} and {settings.tts_speed_max}"
            )
        return v

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        v = (v or "wav").lower().strip()
        if v not in ("wav", "mp3"):
            raise ValueError("format must be 'wav' or 'mp3'")
        return v


class STTResponse(BaseModel):
    text: str
    language: str
    duration_seconds: Optional[float] = None
    model: str


class ModelState(BaseModel):
    enabled: bool
    ready: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    uptime_seconds: float
    models: Dict[str, ModelState]


class VoicesResponse(BaseModel):
    default: str
    voices: List[str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
