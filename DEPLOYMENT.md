# Deployment

The service is a standard ASGI app, so it runs on any Python host. This file
answers the exact deployment questions for **Botkeep Founder Free**
(2 GB RAM / 1.5 vCore / 2 GB storage), then lists portable alternatives.

---

## Read this first: Botkeep is a *bot* host

Botkeep (<https://botkeep.cloud/>) advertises **free hosting for Discord bots**
(Node.js & Python). Its runtime is a long-lived bot process. Before relying on
it for an **HTTP** API, confirm one thing with Botkeep support:

> **Does Botkeep expose a public HTTP port / URL for a Python app?**

Bots normally make *outbound* connections; if there is no inbound HTTP routing,
this API will run but won't be reachable from your websites. The app itself is
host-agnostic, so if the answer is no, use one of the alternatives at the end —
no code changes needed.

Everything below assumes a public port **is** provided and that `$PORT` is set.

---

## Exact Botkeep configuration

| Setting | Value |
| --- | --- |
| **Source** | Connect the GitHub repo `apsideslabs/apsides-voice-api` (or upload a ZIP) |
| **Runtime** | Python |
| **Python version** | **3.12** (pinned in `runtime.txt`; 3.11 also works) |
| **Install / build command** | `pip install -r requirements.txt` |
| **Start command** | `bash start.sh` |
| **Model download command** | `python scripts/download_models.py` (run automatically by `start.sh`) |
| **Health check path** | `/health` (or `/ready` for readiness gating) |

`start.sh` does the whole sequence: download models if missing → locate
`libtranscribe.so` and export `TRANSCRIBE_LIBRARY` → start uvicorn on `$PORT`.

If Botkeep only lets you set a single start command, use this one line instead
of `bash start.sh`:

```bash
python scripts/download_models.py && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
```

### Required environment variables

Only these matter; the rest have safe defaults (see `.env.example`).

| Variable | Value for Botkeep | Why |
| --- | --- | --- |
| `MODEL_DIR` | a **persistent** path, e.g. `/data/models` (or leave default `./models` if the workspace persists) | Keeps ~365 MB of models from re-downloading every restart |
| `PORT` | usually injected by the host | Bind port |
| `STT_BACKEND` | `transcribe_cpp` (default) | Uses the Q8_0 GGUF |
| `ONNX_NUM_THREADS` | `1` | 1.5 vCore — avoid CPU thrash |
| `MAX_CONCURRENCY_TTS` / `_STT` | `1` | 2 GB RAM — one job at a time |
| `CORS_ORIGINS` | your site origins, comma-separated | Lock down who can call it |
| `API_KEY` | a secret you generate (`openssl rand -hex 32`) | Protect your compute |

Set secrets in Botkeep's secret store — **never** commit them.

### Required system dependencies

**None.** The default backend needs no compiler and no system packages:

- `pip install -r requirements.txt` pulls in `transcribe-cpp` **and**
  `transcribe-cpp-native`, which bundles the prebuilt `libtranscribe.so` + ggml
  CPU kernels. Verified: it loads and transcribes with only standard
  glibc/libstdc++ present; the Vulkan module is optional and is not used.
- **ffmpeg** is *optional* — install it only if you want to accept MP3/WebM
  uploads. WAV/FLAC/OGG work with the bundled `soundfile` wheel.
- English G2P uses misaki's **espeak-ng** front-end (no spacy, no torch).
  Do **not** install `misaki[en]` on a 2 GB host — it pulls spacy →
  spacy-curated-transformers → torch + CUDA wheels (**~5.9 GB** installed).
  The engine uses `misaki.en` automatically only if spacy happens to exist.

---

## Where models are stored, and persistence

Everything lives under `MODEL_DIR` (default `./models`, git-ignored):

```
<MODEL_DIR>/
├── kokoro/
│   ├── onnx/model_q8f16.onnx                    ~86 MB
│   ├── voices/*.bin                             ~28 MB (54 voices)
│   └── vocab.json
└── moonshine/
    └── moonshine-streaming-small-Q8_0.gguf      ~189 MB
```

(The native `libtranscribe.so` is **not** here — it ships inside the
`transcribe-cpp-native` pip package. `MODEL_DIR/transcribe/` only appears if the
fallback GitHub bundle was downloaded on a host where pip couldn't fetch the
platform wheel.)

**Persistence:** whether these survive a restart/redeploy depends entirely on
whether Botkeep gives you a persistent volume. If `MODEL_DIR` is on ephemeral
disk, they are re-downloaded on each cold start (`start.sh` handles that
automatically, but it costs a few minutes of bandwidth each time). If Botkeep
offers persistent storage, point `MODEL_DIR` at it and the download happens
once. **Confirm which applies to your plan.**

---

## Expected resource use

| Resource | Estimate | Against Botkeep Founder Free |
| --- | --- | --- |
| Model files on disk | ~303 MB (Kokoro ~114 + GGUF ~189) | ✅ within 2 GB |
| Python deps installed | ~515 MB (measured; onnxruntime + numpy + scipy dominate) | ✅ |
| **Storage total** | **~0.8–0.9 GB** (515 MB deps + ~303 MB models) | ✅ within 2 GB |
| **RAM at runtime** | **~620 MB measured** peak RSS, both models loaded *and* run | ✅ within 2 GB |
| CPU | 1–2 threads | ✅ within 1.5 vCore |

Notes: numbers are estimates from the verified file sizes; actual Python-dep
size varies with wheel versions. If you are tight on RAM, run only one model
(set `TTS_ENABLED=false` or `STT_ENABLED=false`) or use the smaller
`moonshine-streaming-tiny` GGUF (48 MB).

**Performance (measured on a throttled dev pod):** Moonshine Streaming Small
Q8_0 transcribed 11 s of speech in ~13 s, and Kokoro synthesized a 3.2 s phrase
in ~12 s. Both are slower than realtime *here* because the pod CPU is heavily
throttled; a normal CPU is several times faster. Keep `MAX_CONCURRENCY_*=1` and
cap request sizes; raise `ONNX_NUM_THREADS` on hosts with more vCPU.

**Memory note:** the ONNX session runs with `enable_cpu_mem_arena=False` so it
coexists with the STT runtime on 2 GB. Without that, running STT while Kokoro is
resident can OOM.

---

## Verify both models are loaded and working

**1. Readiness** — `/health` reports per-model state:

```bash
curl -s https://YOUR-APP/health | jq '.models'
```
Both `tts` and `stt` should show `"ready": true`. If not, `detail` names the
cause (e.g. a failed download or a missing `libtranscribe.so`).

**2. Full round trip** — synthesize audio, then transcribe it back:

```bash
python scripts/verify_api.py --base https://YOUR-APP
# or, if you set API_KEY:
API_KEY=xxxx python scripts/verify_api.py --base https://YOUR-APP
```

Expected: `health` shows both ready, `tts: OK` with a byte count, and `stt`
returns the spoken text — ending with `RESULT: PASS`.

---

## Final public API endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `POST` | `/tts` | JSON `{text, voice?, speed?, format?}` | `audio/wav` (or `audio/mpeg`) |
| `POST` | `/stt` | multipart `file=<audio>` (+ optional `language`) | JSON `{text, language, duration_seconds, model}` |
| `GET` | `/health` | — | liveness + per-model state |
| `GET` | `/ready` | — | 200/503 readiness |
| `GET` | `/voices` | — | available TTS voices |
| `GET` | `/docs` | — | interactive OpenAPI docs |

Another website calls them like this:

```js
// Text → speech
const audioRes = await fetch("https://YOUR-APP/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": KEY },
  body: JSON.stringify({ text: "Hello!", voice: "af_heart" }),
});
new Audio(URL.createObjectURL(await audioRes.blob())).play();

// Speech → text
const fd = new FormData();
fd.append("file", recordedBlob, "clip.webm");   // WAV recommended; WebM needs ffmpeg
const { text } = await (await fetch("https://YOUR-APP/stt", {
  method: "POST", headers: { "X-API-Key": KEY }, body: fd,
})).json();
```

Remember to add your site's origin to `CORS_ORIGINS`.

---

## Portable alternatives (if Botkeep has no HTTP ingress)

| Host | Free? | Card? | Notes |
| --- | --- | --- | --- |
| Hugging Face Spaces (Docker) | CPU Basic needs a paid plan now; Static is free | no | Not ideal for this CPU service |
| Render (free web service) | 512 MB / 0.1 vCPU | no | Tight RAM — run one model only |
| A small VPS (student credits) | varies | varies | Most reliable persistent public URL |
| Docker anywhere | — | — | `docker build -t apsides-voice-api . && docker run -p 8000:8000 -v $PWD/models:/app/models apsides-voice-api` |

Because the app only needs `$PORT` and a writable `MODEL_DIR`, it moves between
these with no code changes.

---

## Switching STT backend

The default `transcribe_cpp` runs the **Q8_0 GGUF**. To use the `.ort`
packaging of the same model instead:

```bash
pip install -r requirements-moonshine.txt
export STT_BACKEND=moonshine_voice
python scripts/download_models.py --stt
```

Both are the same upstream Moonshine Streaming Small model; only the runtime
packaging differs.
