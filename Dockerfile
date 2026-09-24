# PoC shelf recognition — same stack as local uvicorn demo
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    TRANSFORMERS_CACHE=/data/hf \
    ULTRALYTICS_HOME=/data/ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first (keeps image smaller / works on Docker Desktop Mac+Windows)
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY poc/ poc/
COPY web/ web/
COPY catalog/ catalog/
COPY samples/ samples/
COPY planograms/ planograms/
COPY scripts/docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /app/models /app/out /data/hf /data/ultralytics

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
