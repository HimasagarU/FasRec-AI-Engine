# ── Stage 1: Base with Python + dependencies ──
FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps for PIL/faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (no GPU in container by default)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: App ──
FROM base AS app

WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/styles.csv ./data/styles.csv

# Copy precomputed artifacts (embeddings NOT needed at serve time)
COPY artifacts/precomputed_recs.json ./artifacts/precomputed_recs.json

# Create images directory mount point
RUN mkdir -p ./data/images

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

ENV PORT=8000
CMD uvicorn src.api:app --host 0.0.0.0 --port $PORT --workers 1
