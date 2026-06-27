# syntax=docker/dockerfile:1
# Worker image. Build context is the project root. Installs the FULL agent stack
# (langgraph + torch via sentence-transformers, etc.) — this is the heavy build,
# but only the worker pays for it now.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first for layer caching; the pip cache mount avoids re-downloading
# the large torch wheel on every rebuild.
COPY requirements/base.txt requirements/worker.txt ./requirements/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements/worker.txt

# App code: the worker runs the full agent (shared + RAG + sandbox + agent).
COPY shared ./shared
COPY rag ./rag
COPY sandbox ./sandbox
COPY agent ./agent
COPY worker ./worker

CMD ["python", "-m", "worker.main"]
