# syntax=docker/dockerfile:1
# Worker service image. Build context is the project root (see docker-compose.yml),
# so COPY paths are relative to triage-agent/.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The worker runs the full agent: shared code, RAG, the sandbox runner, the
# agent package (incl. prompt templates), and the worker itself.
COPY shared ./shared
COPY rag ./rag
COPY sandbox ./sandbox
COPY agent ./agent
COPY worker ./worker

CMD ["python", "-m", "worker.main"]
