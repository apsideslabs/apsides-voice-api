"""Domain errors and global exception handlers."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("apsides.voice.errors")


class VoiceError(Exception):
    """Base class for recoverable, client-facing errors."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class ModelNotReadyError(VoiceError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class AudioDecodeError(VoiceError):
    status_code = status.HTTP_400_BAD_REQUEST


class UnsupportedFormatError(VoiceError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class SynthesisError(VoiceError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(VoiceError)
    async def _voice_error(_: Request, exc: VoiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": type(exc).__name__, "detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "ValidationError", "detail": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "HTTPError", "detail": str(exc.detail)},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "detail": "Unexpected server error."},
        )
