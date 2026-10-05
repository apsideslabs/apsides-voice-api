# Security Policy

## Supported versions

The latest release on the `main` branch is supported with security updates.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, email **apsideslabs@gmail.com** with:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept),
- the affected version/commit, and
- any suggested mitigation.

We aim to acknowledge reports within a few business days and will coordinate a fix and disclosure timeline with you.

## Security model and defaults

Apsides Voice API is designed to be safe to self-host, but a few defaults matter:

- **Authentication is opt-in.** If `API_KEY` is unset, `/tts` and `/stt` are **open**. Always set `API_KEY` in production to protect your compute. The comparison is constant-time (`secrets.compare_digest`).
- **CORS defaults to `*`.** Restrict `CORS_ORIGINS` to your own origins in production.
- **No secrets are stored in the repository.** All credentials are read from the environment. `.env` files are git-ignored; only `.env.example` is tracked.
- **Model files are fetched from public sources** (Hugging Face, GitHub releases) at deploy time and cached under `MODEL_DIR`.

## Hardening recommendations

- Put the service behind a reverse proxy / API gateway and terminate TLS there.
- Set a strong `API_KEY` (`openssl rand -hex 32`) and rotate it if exposed.
- Limit `STT_MAX_UPLOAD_MB` and keep `MAX_CONCURRENCY_*` at safe values to bound resource use.
- Keep dependencies updated; pin ranges are intentional (see `requirements.txt`).
- Run the container as a non-root user in production where your platform allows it.

## Dependencies

This project depends on third-party runtimes and model weights. Review their licenses and security posture independently:

- `onnxruntime`, `fastapi`, `uvicorn`, `soundfile`, `transcribe-cpp`
- Kokoro-82M (Apache-2.0), Moonshine Streaming (MIT), transcribe.cpp (MIT)
