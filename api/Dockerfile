# syntax=docker/dockerfile:1
# API gateway image. Build context is the project root (see docker-compose.yml).
# Installs only the LIGHT api dependency set (no torch/langgraph) -> fast builds.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first (before app code) so code changes don't reinstall deps.
# The pip cache mount makes re-downloads instant across builds.
COPY requirements/base.txt requirements/api.txt ./requirements/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements/api.txt

# App code. The API never imports rag.* (/reindex enqueues to the worker), so no
# rag/ is copied here — keeps the image small.
COPY shared ./shared
COPY api ./api

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
