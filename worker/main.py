"""Background worker: an RQ consumer for the ``triage`` queue.

Replaces the Phase-0 heartbeat. Listens on Redis (``settings.redis_url``) and
runs ``worker.handler.handle_triage_job`` for each job.

Worker class is chosen by platform: the classic forking ``Worker`` on POSIX
(production containers, for job isolation), and the non-forking ``SimpleWorker``
on Windows where ``os.fork`` is unavailable (handy for local dev).
"""

from __future__ import annotations

import logging
import os

from redis import Redis
from rq import Queue, SimpleWorker, Worker

from shared.config import settings
from shared.queue import QUEUE_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("triage.worker")


def main() -> None:
    """Connect to Redis and process triage jobs until killed."""
    connection = Redis.from_url(settings.redis_url)
    worker_cls = Worker if hasattr(os, "fork") else SimpleWorker

    logger.info(
        "Worker alive — dry_run=%s enable_live_writes=%s; listening on '%s' queue via %s",
        settings.dry_run,
        settings.enable_live_writes,
        QUEUE_NAME,
        worker_cls.__name__,
    )
    worker = worker_cls([Queue(QUEUE_NAME, connection=connection)], connection=connection)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
