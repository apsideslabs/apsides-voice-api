# Apsides Voice API

A reusable, production-oriented **HTTP AI voice backend** you can connect to any
number of websites and platforms. One FastAPI service, two endpoints:

| Endpoint | Model | Runtime |
| --- | --- | --- |
| `POST /tts` | **Kokoro-82M** (ONNX `model_q8f16`, ~86 MB) | onnxruntime (CPU) |
| `POST /stt` | **Moonshine Streaming Small Q8_0** (123M, ~189 MB GGUF) | transcribe.cpp (ggml CPU) |

Everything runs **CPU-only**, so it fits small hosts (down to ~0.5–1.5 vCPU).
Models are downloaded at deploy time and cached — **no model binaries live in
git**.

- TTS model: <https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX>
- STT (GGUF): <https://huggingface.co/handy-computer/moonshine-streaming-small-gguf>
- STT runtime: <https://github.com/handy-computer/transcribe.cpp>

---

## Project layout

```
apsides-voice-api/
├── main.py                       # app entrypoint (uvicorn main:app)
├── app/
│   ├── config.py                 # env-driven settings (no secrets in code)
│   ├── schemas.py                # request/response validation
│   ├── security.py               # optional X-API-Key auth
│   ├── errors.py                 # domain errors + handlers
│   ├── audio.py                  # decode (soundfile/ffmpeg) + encode (wav/mp3)
│   ├── model_download.py         # fetch + cache all model files
│   ├── registry.py               # loads engines once, reports readiness
│   ├── api.py                    # /tts, /stt, /health, /ready, /voices
│   └── engines/
│       ├── tts.py                # Kokoro ONNX engine
│       └── stt.py                # STT engines (transcribe_cpp | moonshine_voice)
├── scripts/
│   ├── download_models.py        # CLI: fetch + cache models
│   └── verify_api.py             # end-to-end smoke test against a live API
├── requirements.txt
├── requirements-moonshine.txt    # optional alternative STT backend
├── .env.example
├── start.sh                      # download-then-serve (Botkeep/Render/etc.)
├── Procfile / runtime.txt
└── DEPLOYMENT.md
```

---

## Quickstart (local)

```bash
git clone https://github.com/apsideslabs/apsides-voice-api.git
cd apsides-voice-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download models into ./models (Kokoro ~114 MB + Moonshine GGUF ~189 MB)
python scripts/download_models.py

cp .env.example .env          # edit if you want an API key / CORS origins
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs>. Then verify both models:

```bash
python scripts/verify_api.py
```

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
  "version": "1.1.0",
  "uptime_seconds": 42.1,
  "models": {
    "tts": { "enabled": true, "ready": true, "detail": null },
    "stt": { "enabled": true, "ready": true, "detail": null, "backend": "transcribe_cpp" }
  }
}
```

`GET /ready` returns **200** once at least one model is loaded, **503** otherwise.

### `GET /voices` — list TTS voices

### `POST /tts` — text to speech

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

```bash
curl -s -X POST localhost:8000/stt -F 'file=@recording.wav' | jq
```
```json
{ "text": "hello from apsides", "language": "en", "duration_seconds": 2.31, "model": "moonshine-streaming-small-q8:transcribe_cpp" }
```

### Authentication (optional)

Set `API_KEY` to require `X-API-Key` on `/tts` and `/stt`. Empty = open (dev).

---

## Connecting multiple frontends

Set `CORS_ORIGINS` to a comma-separated allow-list:

```
CORS_ORIGINS=https://linguilo.app,https://jigyasa.example,https://luitra.example
```

```js
const res = await fetch("https://your-api-host/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": KEY },
  body: JSON.stringify({ text: "Namaste!", voice: "af_heart" }),
});
new Audio(URL.createObjectURL(await res.blob())).play();
```

---

## Configuration

All settings are environment variables — see `.env.example`. Key ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_DIR` | `./models` | Where models are cached (use a persistent volume) |
| `AUTO_DOWNLOAD_MODELS` | `true` | Fetch missing models during startup |
| `PORT` | `8000` | HTTP port (hosts inject `$PORT`) |
| `STT_BACKEND` | `transcribe_cpp` | `transcribe_cpp` (GGUF q8_0) or `moonshine_voice` |
| `TRANSCRIBE_VERSION` | `0.3.0` | Native bundle version — keep in sync with `transcribe-cpp` pin |
| `CORS_ORIGINS` | `*` | Allowed origins (comma-separated) |
| `API_KEY` | *(unset)* | If set, protects `/tts` and `/stt` |
| `ONNX_NUM_THREADS` | `1` | Keep at 1 on small vCPU hosts |
| `MAX_CONCURRENCY_TTS/STT` | `1` | Simultaneous inference jobs |

---

## Design notes

- **Resource-bounded:** single-threaded ONNX, a semaphore per endpoint, and
  blocking inference pushed to a threadpool.
- **Two STT backends:** `transcribe_cpp` runs the Q8_0 GGUF via the ggml CPU
  runtime (default); `moonshine_voice` runs the `.ort` packaging of the same
  model. Pick with `STT_BACKEND`.
- **Graceful degradation:** a missing model never stops the API booting;
  `/health` shows the reason.
- **No secrets in the repo:** the API key comes from the environment only.
- **No compiler needed:** the native runtime ships inside the
  `transcribe-cpp-native` wheel that `pip install transcribe-cpp` pulls in.
- **Fits a 2 GB host:** deps ~515 MB, models ~303 MB, peak RAM ~620 MB. English
  G2P uses misaki's espeak-ng front-end — deliberately *not* `misaki[en]`, which
  would drag in spacy + torch + CUDA wheels (~5.9 GB).

## License

MIT — see [LICENSE](LICENSE). Model weights carry their own licenses
(Kokoro-82M: Apache-2.0; Moonshine Streaming: MIT; transcribe.cpp: MIT).
