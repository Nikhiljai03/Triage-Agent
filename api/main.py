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

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from api.webhook import parse_issue_event, verify_signature
from shared.config import settings
from shared.queue import enqueue_triage
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
async def reindex(background: BackgroundTasks) -> JSONResponse:
    """Kick off RAG ingestion for the target repo in the background (non-blocking)."""
    background.add_task(_run_reindex, settings.target_repo)
    return JSONResponse(
        {"status": "reindex_started", "repo": settings.target_repo}, status_code=202
    )


def _run_reindex(repo: str) -> None:
    """Thin wrapper around Phase-1 ingestion; imported lazily (heavy deps)."""
    try:
        from rag.ingest import ingest_repo

        logger.info("Reindex starting for %s", repo)
        ingest_repo(repo)
        logger.info("Reindex finished for %s", repo)
    except Exception:  # noqa: BLE001 — background task must never escape unlogged.
        logger.exception("Reindex failed for %s", repo)


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
