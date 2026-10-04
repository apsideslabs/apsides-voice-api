"""Apsides Voice API — application entrypoint.

A reusable, production-oriented HTTP backend that exposes a single voice API
for multiple websites/platforms:

    * POST /tts   -> Kokoro-82M (ONNX, q8f16) text-to-speech
    * POST /stt   -> Moonshine Streaming Small (q8) English speech-to-text
    * GET  /health, /ready, /voices, /  -> health & introspection

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000

Run in production (respects $PORT):
    python main.py
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from app.api import router
from app.config import settings
from app.errors import register_exception_handlers
from app.registry import registry


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


_configure_logging(settings.log_level)
logger = logging.getLogger("apsides.voice")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Reusable HTTP AI voice backend: Kokoro TTS + Moonshine STT, "
            "CPU-only, designed for constrained hosts."
        ),
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # --- Middleware -------------------------------------------------------
    # CORS is configured from CORS_ORIGINS so any number of frontends can
    # call this API. Use "*" for development; list explicit origins in prod.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=settings.allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id", "X-Inference-Ms"],
        max_age=86400,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    register_exception_handlers(app)
    app.include_router(router)
    return app


from contextlib import asynccontextmanager  # noqa: E402  (kept near usage)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    logger.info("Starting %s v%s", settings.app_name, settings.version)
    registry.load_all(settings)
    for name, state in registry.status().items():
        logger.info("model '%s': %s", name, state)
    yield
    registry.shutdown()
    logger.info("Shutdown complete")


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.effective_port,
        log_level=settings.log_level.lower(),
    )
