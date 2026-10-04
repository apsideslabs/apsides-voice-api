"""HTTP API routes."""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.audio import decode_audio, encode_audio
from app.config import settings
from app.errors import ModelNotReadyError, UnsupportedFormatError
from app.registry import registry
from app.schemas import (
    HealthResponse,
    ModelState,
    STTResponse,
    TTSRequest,
    VoicesResponse,
)
from app.security import require_api_key

logger = logging.getLogger("apsides.voice.api")
router = APIRouter()

# Bound concurrent inference so a constrained host is never oversubscribed.
_tts_sem = asyncio.Semaphore(max(1, settings.max_concurrency_tts))
_stt_sem = asyncio.Semaphore(max(1, settings.max_concurrency_stt))


def _state(model: str) -> ModelState:
    s = registry.status().get(model, {"enabled": False, "ready": False, "detail": None})
    return ModelState(
        enabled=bool(s.get("enabled")),
        ready=bool(s.get("ready")),
        detail=s.get("detail"),
    )


@router.get("/", summary="Service info")
async def root() -> dict:
    return {
        "app": settings.app_name,
        "version": settings.version,
        "endpoints": {
            "tts": "POST /tts",
            "stt": "POST /stt",
            "health": "GET /health",
            "ready": "GET /ready",
            "voices": "GET /voices",
            "docs": "GET /docs",
        },
        "auth_required": bool(settings.api_key),
    }


@router.get("/health", response_model=HealthResponse, summary="Liveness + model state")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.version,
        uptime_seconds=registry.uptime(),
        models={"tts": _state("tts"), "stt": _state("stt")},
    )


@router.get("/ready", summary="Readiness (503 until a model is loaded)")
async def ready() -> JSONResponse:
    ok = registry.any_ready()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"ready": ok, "models": registry.status()},
    )


@router.get("/voices", response_model=VoicesResponse, summary="List TTS voices")
async def voices() -> VoicesResponse:
    if registry.tts is None:
        return VoicesResponse(default=settings.tts_default_voice, voices=[])
    return VoicesResponse(
        default=settings.tts_default_voice,
        voices=registry.tts.list_voices(),
    )


@router.post(
    "/tts",
    summary="Text-to-speech (Kokoro-82M ONNX)",
    dependencies=[Depends(require_api_key)],
    responses={200: {"content": {"audio/wav": {}, "audio/mpeg": {}}}},
)
async def tts(payload: TTSRequest) -> Response:
    if registry.tts is None:
        raise ModelNotReadyError("TTS is disabled or not loaded on this instance.")

    started = time.time()
    async with _tts_sem:
        samples = await run_in_threadpool(
            registry.tts.synthesize, payload.text, payload.voice, payload.speed
        )
    audio, media_type = encode_audio(samples, 24000, payload.format)
    elapsed_ms = int((time.time() - started) * 1000)
    logger.info(
        "tts ok chars=%d voice=%s fmt=%s ms=%d",
        len(payload.text), payload.voice or settings.tts_default_voice, payload.format, elapsed_ms,
    )
    return Response(
        content=audio,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="speech.{payload.format}"',
            "X-Inference-Ms": str(elapsed_ms),
            "Cache-Control": "no-store",
        },
    )


@router.post(
    "/stt",
    response_model=STTResponse,
    summary="Speech-to-text (Moonshine Streaming Small)",
    dependencies=[Depends(require_api_key)],
)
async def stt(file: UploadFile = File(...), language: str | None = Form(default=None)) -> STTResponse:
    if registry.stt is None:
        raise ModelNotReadyError("STT is disabled or not loaded on this instance.")

    data = await file.read()
    max_bytes = settings.stt_max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise UnsupportedFormatError(
            f"Upload too large ({len(data)} bytes); limit is {settings.stt_max_upload_mb} MB."
        )

    started = time.time()
    samples, sr = await run_in_threadpool(decode_audio, data)
    duration = round(len(samples) / sr, 3) if sr else None
    async with _stt_sem:
        text = await run_in_threadpool(registry.stt.transcribe, samples, sr)
    elapsed_ms = int((time.time() - started) * 1000)
    logger.info("stt ok sr=%d dur=%s ms=%d", sr, duration, elapsed_ms)

    return STTResponse(
        text=text,
        language=language or settings.stt_language,
        duration_seconds=duration,
        model=f"moonshine-{settings.stt_model_arch}",
    )
