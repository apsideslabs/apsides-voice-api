# Deployment

The service is a standard ASGI app (`uvicorn main:app`), so it runs on any
Python host. Below: the target host (Botkeep) plus portable fallbacks.

---

## Target: Botkeep — *read this first*

Botkeep (<https://botkeep.cloud/>) advertises itself as **free hosting for
Discord bots** (Node.js & Python). Its runtime is a long-lived bot process with
resource limits and a protected environment. Two things to verify before you
rely on it for an **HTTP** API:

1. **Public HTTP ingress.** The product is framed around bots that make
   *outbound* connections (e.g. to Discord), not around exposing a public HTTP
   port. If Botkeep does not route inbound HTTP to your process, this API will
   run but **not be reachable from websites**. Confirm with Botkeep support
   whether a public URL/port is provided for Python apps.
2. **Limits.** The public beta lists **512 MB RAM / 0.5 vCPU / 1 GB storage per
   bot**. Kokoro + Moonshine together plus Python and onnxruntime can exceed
   512 MB RAM and ~300 MB of installed dependencies + ~190 MB of models. On the
   stated *Founder* allocation (2 GB RAM / 1.5 vCore / 2 GB storage) you have
   room; on the 512 MB tier you likely do not. Check which applies to you.

If Botkeep does expose an HTTP port, use the settings below. If not, use one of
the portable hosts further down.

### Botkeep setup

- **Source:** connect this GitHub repository (Botkeep supports GitHub + ZIP).
- **Runtime:** Python.
- **Start command:**
  ```bash
  python scripts/download_models.py && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
  ```
  (or `bash start.sh`, which does the same). Downloading on first boot keeps the
  ~190 MB of models out of git; point `MODEL_DIR` at persistent storage so they
  are not re-fetched every restart.
- **Environment variables:** copy the values from `.env.example`. At minimum set
  `MODEL_DIR` (persistent path), `PORT`, `CORS_ORIGINS`, and `API_KEY`.
  **Never paste secrets into the repo** — use the host's secret/env store.

---

## Portable hosts (if Botkeep has no HTTP ingress)

| Host | Free? | Card? | Notes |
| --- | --- | --- | --- |
| Hugging Face Spaces (Gradio/ZeroGPU) | free GPU, 2 Spaces | no | Best free GPU path; wrap the API in a Gradio app or use a Static Space + API proxy |
| Render (free web service) | 512 MB / 0.1 vCPU | no | Sleeps after 15 min idle; tight RAM — expect to disable one model |
| A small VPS (₹0 via student credits) | varies | varies | Most reliable for a persistent public URL |
| Your own machine + Cloudflare Tunnel | free | no | Great for dev/demo |

Because the app only needs `$PORT` and a writable `MODEL_DIR`, it moves between
these with no code changes.

### Generic start (any host)

```bash
pip install -r requirements.txt
python scripts/download_models.py        # once; cached under MODEL_DIR
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
```

### Docker

```bash
docker build -t apsides-voice-api .
docker run -p 8000:8000 -v $PWD/models:/app/models apsides-voice-api
```

---

## Sizing & tuning for constrained hosts

| Knob | Recommended on ≤1.5 vCPU | Effect |
| --- | --- | --- |
| `ONNX_NUM_THREADS` | `1` | Avoids CPU thrash on small vCPU shares |
| `MAX_CONCURRENCY_TTS` / `_STT` | `1` | One job at a time; prevents OOM |
| `STT_MODEL_ARCH` | `small_streaming` (or `tiny_streaming`) | `tiny_streaming` halves memory if RAM is tight |
| `TTS_ENABLED` / `STT_ENABLED` | disable one if RAM-limited | Run a single model on 512 MB tiers |

Model sizes to budget for: Kokoro `model_q8f16.onnx` ≈ **86 MB** + voices
≈ **28 MB**; Moonshine small-streaming q8 ≈ **130 MB**; plus installed Python
deps. Keep `MODEL_DIR` on a persistent volume so this is a one-time download.

---

## Health checks

- Liveness: `GET /health` → always `200` while the process runs.
- Readiness: `GET /ready` → `200` when a model is loaded, `503` otherwise.

Wire your platform's probe to `/ready` so traffic only arrives once models are
in memory.
