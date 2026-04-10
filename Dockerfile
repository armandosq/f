# ── Base: Python slim (CPU). Railway no tiene GPU de todas formas.
# Si en el futuro necesitás GPU, volvé a nvidia/cuda pero usá una imagen
# más chica: nvidia/cuda:12.3.1-base-ubuntu22.04
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── Sistema: FFmpeg + dependencias ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── App ────────────────────────────────────────────────────────
COPY app.py .

# Railway inyecta $PORT automáticamente; exponemos el default
EXPOSE 7860

CMD ["python", "app.py"]
