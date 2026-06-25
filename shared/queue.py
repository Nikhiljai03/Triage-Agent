"""Redis + RQ job queue and the ``enqueue_triage`` entrypoint.

The API enqueues a triage job here; the worker (``worker.main``) consumes it and
runs ``worker.handler.handle_triage_job``. A :class:`TriageRun` is created in the
run store with status ``"queued"`` *at enqueue time* so the dashboard shows the
run immediately, before the worker has even picked it up.

The job is referenced by string path (``"worker.handler.handle_triage_job"``) so
the API process never has to import worker code.
"""

from __future__ import annotations

import logging
import uuid

from redis import Redis
from rq import Queue

from shared.config import settings
from shared.run_store import get_run_store
from shared.schemas import IssueEvent, TriageRun, TriageStep

logger = logging.getLogger("triage.queue")

QUEUE_NAME = "triage"
JOB_FUNC = "worker.handler.handle_triage_job"

# RQ needs a raw (bytes) connection — do NOT use decode_responses here.
_redis_singleton: Redis | None = None
_queue_singleton: Queue | None = None


def get_redis() -> Redis:
    global _redis_singleton
    if _redis_singleton is None:
        _redis_singleton = Redis.from_url(settings.redis_url)
    return _redis_singleton


def get_queue() -> Queue:
    """Return the process-wide RQ queue (created lazily)."""
    global _queue_singleton
    if _queue_singleton is None:
        _queue_singleton = Queue(QUEUE_NAME, connection=get_redis())
    return _queue_singleton


def set_queue(queue: Queue) -> None:
    """Override the queue singleton — used by tests to inject a fake queue."""
    global _queue_singleton
    _queue_singleton = queue


def enqueue_triage(event: IssueEvent) -> str:
    """Create a queued :class:`TriageRun`, enqueue the job, and return its ``run_id``.

    The ``run_id`` doubles as the RQ ``job_id`` so a run and its job share one id.
    """
    run_id = uuid.uuid4().hex
    run = TriageRun(
        run_id=run_id,
        repo=event.repo,
        issue_number=event.issue_number,
        status="queued",
        steps=[
            TriageStep(name="queued", detail=f"job enqueued for {event.repo}#{event.issue_number}")
        ],
    )
    get_run_store().create(run)

    # Reserved kwargs (job_id) are consumed by RQ; the function receives (event, run_id).
    get_queue().enqueue(JOB_FUNC, event.model_dump(mode="json"), run_id, job_id=run_id)
    logger.info("Enqueued triage run %s for %s#%s", run_id, event.repo, event.issue_number)
    return run_id
