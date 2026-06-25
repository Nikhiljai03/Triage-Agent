"""Background worker for the Triage Agent.

Phase 0: a heartbeat loop that proves the container stays up and reads the same
shared config as the API. The real agent pipeline (RAG -> sandbox repro ->
classify -> draft PR) is wired in starting Phase 4.
"""

from __future__ import annotations

import logging
import time

from shared.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("triage.worker")

# Seconds between heartbeats. Kept short so `docker compose logs` shows life
# quickly; real job polling replaces this loop in a later phase.
HEARTBEAT_SECONDS = 15


def main() -> None:
    """Log an 'alive' line, then heartbeat forever so the container stays up."""
    logger.info(
        "Worker alive — dry_run=%s enable_live_writes=%s",
        settings.dry_run,
        settings.enable_live_writes,
    )
    while True:
        logger.info("heartbeat — idle, waiting for jobs (Phase 0 stub)")
        time.sleep(HEARTBEAT_SECONDS)


if __name__ == "__main__":
    main()
