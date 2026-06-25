"""FastAPI gateway for the Triage Agent.

Phase 0: exposes only a ``/health`` liveness probe so the service can be wired
into docker-compose and verified end to end. The GitHub webhook receiver and
the job-enqueue path are added in a later phase.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("triage.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log safety-relevant config on boot (and teardown hook for later phases)."""
    logger.info(
        "API alive — dry_run=%s target_repo=%s",
        settings.dry_run,
        settings.target_repo,
    )
    yield


app = FastAPI(
    title="Triage Agent API",
    description="Autonomous GitHub issue-triage gateway (Phase 0 skeleton).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe used by docker-compose and uptime checks."""
    return {"status": "ok"}
