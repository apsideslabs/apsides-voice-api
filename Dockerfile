# Apsides Voice API — CPU-only container image.
FROM python:3.12-slim

# ffmpeg enables MP3/WebM upload decoding (optional but recommended).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_DIR=/app/models \
    PORT=8000 \
    ONNX_NUM_THREADS=1

# Download models at build time so the image is self-contained.
# (Remove this line and mount a volume if you prefer download-at-boot.)
RUN python scripts/download_models.py || true

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
