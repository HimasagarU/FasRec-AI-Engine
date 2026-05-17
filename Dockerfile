FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS app

WORKDIR /app

COPY backend/src/ ./src/
COPY backend/data/styles.csv ./data/styles.csv
COPY backend/artifacts/parsed_products.csv ./artifacts/parsed_products.csv
COPY backend/artifacts/precomputed_recs.json ./artifacts/precomputed_recs.json
RUN mkdir -p ./data/images

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

ENV PORT=8000
CMD uvicorn src.api:app --host 0.0.0.0 --port $PORT --workers 1
