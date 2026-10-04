#!/usr/bin/env python3
"""End-to-end smoke test for a running Apsides Voice API.

Checks that /health reports both models ready, then round-trips real audio:
TTS synthesizes a phrase, and STT transcribes that same audio back.

Usage:
    python scripts/verify_api.py                          # http://localhost:8000
    python scripts/verify_api.py --base https://my.host   # deployed instance
    API_KEY=xxxx python scripts/verify_api.py             # if auth is enabled
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request


def _post(base: str, path: str, data: bytes, headers: dict) -> bytes:
    req = urllib.request.Request(base + path, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _get(base: str, path: str, headers: dict) -> bytes:
    req = urllib.request.Request(base + path, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE_URL", "http://localhost:8000"))
    args = ap.parse_args()
    base = args.base.rstrip("/")

    headers = {}
    if os.environ.get("API_KEY"):
        headers["X-API-Key"] = os.environ["API_KEY"]

    ok = True

    # 1) health ---------------------------------------------------------
    health = json.loads(_get(base, "/health", headers))
    print("health:", json.dumps(health.get("models", {}), indent=2))
    for name in ("tts", "stt"):
        st = health["models"].get(name, {})
        if not st.get("ready"):
            print(f"  !! {name} NOT ready: {st.get('detail')}")
            ok = False

    # 2) TTS round trip -------------------------------------------------
    phrase = "The quick brown fox jumps over the lazy dog."
    try:
        body = json.dumps({"text": phrase, "voice": "af_heart", "format": "wav"}).encode()
        audio = _post(base, "/tts", body, {**headers, "Content-Type": "application/json"})
        print(f"tts: OK, {len(audio)} bytes of WAV")
    except Exception as exc:  # noqa: BLE001
        print(f"tts: FAILED -> {exc}")
        return 1

    # 3) STT round trip (multipart form) --------------------------------
    boundary = "----apsidesverify"
    parts = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    try:
        resp = _post(
            base, "/stt", parts,
            {**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        result = json.loads(resp)
        print("stt:", json.dumps(result, indent=2))
        if not result.get("text"):
            print("  !! stt returned empty text")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"stt: FAILED -> {exc}")
        return 1

    print("\nRESULT:", "PASS" if ok else "DEGRADED (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
