"""Redis-backed persistence for :class:`TriageRun` records.

Storage layout:

* ``triage:runs``   — a Redis hash mapping ``run_id -> TriageRun`` (JSON).
* ``triage:recent`` — a capped list of ``run_id``s, newest first, for the dashboard.

This is the single source of truth the dashboard and ``/status`` read. The store
uses a ``decode_responses=True`` connection (strings, not bytes) so it must NOT
be shared with the RQ queue connection (RQ needs raw bytes).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from redis import Redis

from shared.config import settings
from shared.schemas import TriageRun, TriageStep

logger = logging.getLogger("triage.run_store")

_RUNS_HASH = "triage:runs"
_RECENT_LIST = "triage:recent"
_RECENT_MAX = 200  # cap the recent index so it can't grow unbounded


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RunStore:
    """Create/read/update :class:`TriageRun` records in Redis."""

    def __init__(self, connection: Redis | None = None) -> None:
        self._redis = (
            connection
            if connection is not None
            else Redis.from_url(settings.redis_url, decode_responses=True)
        )

    # -- writes ------------------------------------------------------------
    def create(self, run: TriageRun) -> TriageRun:
        """Persist a new run and push it onto the (capped) recent index."""
        self._redis.hset(_RUNS_HASH, run.run_id, run.model_dump_json())
        self._redis.lpush(_RECENT_LIST, run.run_id)
        self._redis.ltrim(_RECENT_LIST, 0, _RECENT_MAX - 1)
        return run

    def update(self, run_id: str, **fields: object) -> TriageRun | None:
        """Patch fields on an existing run (always bumps ``updated_at``)."""
        run = self.get(run_id)
        if run is None:
            logger.warning("update() for unknown run_id %s", run_id)
            return None
        for key, value in fields.items():
            setattr(run, key, value)
        run.updated_at = _utcnow()
        self._redis.hset(_RUNS_HASH, run_id, run.model_dump_json())
        return run

    def append_step(self, run_id: str, name: str, detail: str = "") -> TriageRun | None:
        """Append one reasoning-trace step and persist."""
        run = self.get(run_id)
        if run is None:
            logger.warning("append_step() for unknown run_id %s", run_id)
            return None
        run.steps.append(TriageStep(name=name, detail=detail))
        run.updated_at = _utcnow()
        self._redis.hset(_RUNS_HASH, run_id, run.model_dump_json())
        return run

    # -- reads -------------------------------------------------------------
    def get(self, run_id: str) -> TriageRun | None:
        raw = self._redis.hget(_RUNS_HASH, run_id)
        return TriageRun.model_validate_json(raw) if raw else None

    def list_recent(self, limit: int = 20) -> list[TriageRun]:
        """Return up to ``limit`` most-recent runs, newest first."""
        run_ids = self._redis.lrange(_RECENT_LIST, 0, limit - 1)
        if not run_ids:
            return []
        raws = self._redis.hmget(_RUNS_HASH, run_ids)
        return [TriageRun.model_validate_json(raw) for raw in raws if raw]


# --- module-level singleton (so API + queue share one store) ----------------
_store_singleton: RunStore | None = None


def get_run_store() -> RunStore:
    """Return the process-wide :class:`RunStore` (created lazily)."""
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = RunStore()
    return _store_singleton


def set_run_store(store: RunStore) -> None:
    """Override the singleton — used by tests to inject a fakeredis-backed store."""
    global _store_singleton
    _store_singleton = store
