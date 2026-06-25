# syntax=docker/dockerfile:1
# API service image. Build context is the project root (see docker-compose.yml),
# so COPY paths are relative to triage-agent/.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy what the API needs: shared code, the api package (incl. templates), and
# rag/ (the /reindex endpoint reuses the Phase-1 ingestion pipeline).
COPY shared ./shared
COPY rag ./rag
COPY api ./api

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
