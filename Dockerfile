# ── Base: Ubuntu con soporte CUDA (para GPU) ──────────────────
# Si Railway no tiene GPU, esta imagen igual funciona en CPU.
# Cambiá a "ubuntu:22.04" si solo usás CPU para imagen más liviana.
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── Sistema: Python + FFmpeg + dependencias ───────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    ffmpeg \
    curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Alias python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# ── Python deps ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── App ────────────────────────────────────────────────────────
COPY app.py .

# Railway inyecta $PORT automáticamente
EXPOSE 7860

CMD ["python", "app.py"]
