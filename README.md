# Apsides Voice API

A reusable, production-oriented **HTTP AI voice backend** you can connect to any
number of websites and platforms. One FastAPI service, two endpoints:

| Endpoint | Model | Purpose |
| --- | --- | --- |
| `POST /tts` | **Kokoro-82M** (ONNX `model_q8f16`, ~86 MB) | Text → speech |
| `POST /stt` | **Moonshine Streaming Small** (q8, 123M, MIT) | English speech → text |

Everything runs **CPU-only**, so it fits small hosts (down to ~0.5–1.5 vCPU).
Models are downloaded at deploy time and cached — **no model files live in git**.

- TTS model: <https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX>
- STT project: <https://github.com/moonshine-ai/moonshine>

---

## Project layout

```
apsides-voice-api/
├── main.py                     # app entrypoint (uvicorn main:app)
├── app/
│   ├── config.py               # env-driven settings (no secrets in code)
│   ├── schemas.py              # request/response validation
│   ├── security.py             # optional X-API-Key auth
│   ├── errors.py               # domain errors + handlers
│   ├── audio.py                # decode (soundfile/ffmpeg) + encode (wav/mp3)
│   ├── registry.py             # loads engines once, reports readiness
│   ├── api.py                  # /tts, /stt, /health, /ready, /voices
│   └── engines/
│       ├── tts.py              # Kokoro ONNX engine
│       └── stt.py              # Moonshine engine
├── scripts/download_models.py  # fetch + cache models
├── requirements.txt
├── .env.example
├── DEPLOYMENT.md
└── start.sh
```

---

## Quickstart (local)

```bash
git clone https://github.com/apsideslabs/apsides-voice-api.git
cd apsides-voice-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download models into ./models (Kokoro ~86 MB + voices, Moonshine cache)
python scripts/download_models.py

cp .env.example .env          # edit if you want an API key / CORS origins
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> for interactive API docs.

> **ffmpeg (optional):** `soundfile` handles WAV/FLAC/OGG. To accept MP3/WebM
> uploads (common from browsers' `MediaRecorder`) install `ffmpeg` on the host.

---

## API

### `GET /health` — liveness + model state

```bash
curl -s localhost:8000/health | jq
```
```json
{
  "status": "ok",
  "app": "apsides-voice-api",
  "version": "1.0.0",
  "uptime_seconds": 42.1,
  "models": {
    "tts": { "enabled": true, "ready": true, "detail": null },
    "stt": { "enabled": true, "ready": true, "detail": null }
  }
}
```

`GET /ready` returns **200** once at least one model is loaded, **503** otherwise —
point your platform's readiness probe at it.

### `GET /voices` — list TTS voices

```bash
curl -s localhost:8000/voices | jq
```

### `POST /tts` — text to speech

Request (JSON), response is raw audio (`audio/wav` by default).

```bash
curl -s -X POST localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Apsides.","voice":"af_heart","speed":1.0,"format":"wav"}' \
  --output speech.wav
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `text` | string | — | Required; ≤ `TTS_MAX_CHARS` |
| `voice` | string | `af_heart` | Any id from `/voices` (`a*`=US, `b*`=UK) |
| `speed` | float | `1.0` | Between `TTS_SPEED_MIN`/`MAX` |
| `format` | string | `wav` | `wav` or `mp3` (mp3 needs ffmpeg) |

### `POST /stt` — speech to text

Multipart form upload. Field name is `file` (optional `language`).

```bash
curl -s -X POST localhost:8000/stt \
  -F 'file=@recording.wav' | jq
```
```json
{ "text": "hello from apsides", "language": "en", "duration_seconds": 2.31, "model": "moonshine-small_streaming" }
```

### Authentication (optional)

Set `API_KEY` in the environment to require the header on `/tts` and `/stt`:

```bash
curl -s -X POST localhost:8000/tts -H "X-API-Key: $API_KEY" ...
```

If `API_KEY` is empty the API is open (fine for local dev; set it in production).

---

## Connecting multiple frontends

CORS is driven by `CORS_ORIGINS`. Set it to a comma-separated allow-list so any
number of sites can call the API:

```
CORS_ORIGINS=https://linguilo.app,https://jigyasa.example,https://luitra.example
```

Example browser call:

```js
const res = await fetch("https://your-api-host/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": KEY },
  body: JSON.stringify({ text: "Namaste!", voice: "af_heart" }),
});
const url = URL.createObjectURL(await res.blob());
new Audio(url).play();
```

---

## Configuration

All settings are environment variables — see `.env.example` for the full list.
Key ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_DIR` | `./models` | Where models are cached (use a persistent volume) |
| `PORT` | `8000` | HTTP port (hosts inject `$PORT`) |
| `CORS_ORIGINS` | `*` | Allowed origins (comma-separated) |
| `API_KEY` | *(unset)* | If set, protects `/tts` and `/stt` |
| `ONNX_NUM_THREADS` | `1` | Keep at 1 on small vCPU hosts |
| `MAX_CONCURRENCY_TTS/STT` | `1` | Simultaneous inference jobs |
| `STT_MODEL_ARCH` | `small_streaming` | tiny / tiny_streaming / base / small_streaming / medium_streaming |

---

## Design notes

- **Resource-bounded:** single-threaded ONNX, a semaphore per endpoint, and
  blocking inference pushed to a threadpool so the event loop stays responsive.
- **Graceful degradation:** if a model is missing the API still boots; the other
  endpoint keeps working and `/health` shows why.
- **No secrets in the repo:** the API key is read from the environment only.
- **Long text** is chunked by sentence so phoneme sequences stay inside the
  model's 512-token context.

## License

MIT — see [LICENSE](LICENSE). The Kokoro and Moonshine model weights carry their
own licenses (Kokoro-82M: Apache-2.0; Moonshine: MIT).
