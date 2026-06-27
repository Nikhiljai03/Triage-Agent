"""FastAPI gateway for the Triage Agent.

Phase 2 makes the system event-driven. Routes:

* ``GET  /health``        — liveness probe (Phase 0).
* ``POST /webhook``       — verify signature, normalize, enqueue a triage job.
* ``GET  /status``        — recent triage runs (JSON).
* ``GET  /status/{id}``   — one run with its full step trace.
* ``POST /reindex``       — kick off RAG ingestion in the background.
* ``GET  /``              — server-rendered dashboard.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from api.webhook import parse_issue_event, verify_signature
from shared.config import settings
from shared.queue import enqueue_reindex, enqueue_triage
from shared.run_store import get_run_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("triage.api")

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log safety-relevant config and registered routes on boot."""
    logger.info("API alive — dry_run=%s target_repo=%s", settings.dry_run, settings.target_repo)
    routes = sorted(r.path for r in app.routes if getattr(r, "path", "").startswith("/"))
    logger.info("Routes registered: %s", ", ".join(routes))
    yield


app = FastAPI(
    title="Triage Agent API",
    description="Autonomous GitHub issue-triage gateway.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe used by docker-compose and uptime checks."""
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    """Receive a GitHub issues webhook: verify -> normalize -> enqueue."""
    body = await request.body()
    if not verify_signature(body, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=401, detail="invalid or missing signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc

    event = parse_issue_event(payload)
    if event is None:
        return JSONResponse({"status": "ignored"}, status_code=200)

    run_id = enqueue_triage(event)
    return JSONResponse({"status": "queued", "run_id": run_id}, status_code=202)


@app.get("/status")
async def status_list() -> list[dict]:
    """Recent triage runs, newest first."""
    return [run.model_dump(mode="json") for run in get_run_store().list_recent()]


@app.get("/status/{run_id}")
async def status_one(run_id: str) -> dict:
    """A single run, including its full reasoning-trace steps."""
    run = get_run_store().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run.model_dump(mode="json")


@app.post("/reindex")
async def reindex() -> JSONResponse:
    """Enqueue a RAG reindex for the target repo (runs in the worker, non-blocking).

    The API stays light by NOT importing ``rag.*`` — it just drops a job on the
    queue; the worker (which carries the heavy RAG deps) does the ingestion.
    """
    job_id = enqueue_reindex(settings.target_repo)
    return JSONResponse(
        {"status": "reindex_enqueued", "repo": settings.target_repo, "job_id": job_id},
        status_code=202,
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Server-rendered dashboard of recent runs."""
    try:
        runs = get_run_store().list_recent()
    except Exception:  # noqa: BLE001 — render an empty board if Redis is unreachable.
        logger.exception("Dashboard could not read the run store")
        runs = []
    return _TEMPLATES.TemplateResponse(
        request, "dashboard.html", {"runs": runs, "repo": settings.target_repo}
    )
