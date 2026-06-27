"""Reindex job handler — runs RAG ingestion inside the worker.

The API's ``/reindex`` endpoint enqueues this (via ``shared.queue.enqueue_reindex``)
instead of importing ``rag.*`` in-process, so the API image can stay light
(no torch). The Phase-1 ingestion logic itself is unchanged.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("triage.worker.reindex")


def handle_reindex_job(repo: str) -> None:
    """Ingest a repo's issues into the vector store. Never crashes the worker."""
    from rag.ingest import ingest_repo

    try:
        logger.info("Reindex starting for %s", repo)
        count = ingest_repo(repo)
        logger.info("Reindex finished for %s: %d vectors upserted", repo, count)
    except Exception:  # noqa: BLE001 — a failed reindex must not take down the worker.
        logger.exception("Reindex failed for %s", repo)
