"""Optional API-key authentication.

The key is read from the environment (API_KEY) and never stored in code or
git. When API_KEY is unset, the API is open (useful for local dev / public
demos); set it in production to protect your compute.
"""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency. No-op unless API_KEY is configured."""
    if not settings.api_key:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
