# Apsides Voice API — CPU-only container image.
FROM python:3.12-slim

# ffmpeg enables MP3/WebM upload decoding (optional but recommended).
# libsndfile1 is needed by soundfile. ca-certificates for model downloads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_DIR=/app/models \
    PORT=8000 \
    ONNX_NUM_THREADS=1 \
    STT_BACKEND=transcribe_cpp \
    AUTO_DOWNLOAD_MODELS=false

# Download models at build time so the image is self-contained.
# (Remove this line and mount a volume if you prefer download-at-boot.)
RUN python scripts/download_models.py || true

EXPOSE 8000
CMD ["bash", "start.sh"]
