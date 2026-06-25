"""Triage job handler (Phase 2 stub).

This is the function RQ invokes for each queued job. For now it just records a
placeholder decision and a few reasoning steps to the run store, proving the
event pipeline end to end. The real agent replaces the stub in Phase 4.

It must NEVER raise out of the worker process: any failure is caught and
recorded on the run as an ``error`` step + status.
"""

from __future__ import annotations

import logging

from shared.run_store import get_run_store
from shared.schemas import IssueEvent

logger = logging.getLogger("triage.worker.handler")


def handle_triage_job(event_dict: dict, run_id: str) -> None:
    """Process one triage job: mark running, record stub steps, mark done.

    Args:
        event_dict: a JSON-serialized :class:`IssueEvent`.
        run_id: the run to update (also the RQ job id).
    """
    store = get_run_store()
    try:
        event = IssueEvent.model_validate(event_dict)
        store.update(run_id, status="running")
        store.append_step(
            run_id,
            "received",
            f"issue #{event.issue_number} '{event.title}' (action={event.action})",
        )
        store.append_step(run_id, "classify", "(agent not implemented yet)")

        # TODO(Phase 4): replace this stub with the LangGraph triage agent
        # (RAG duplicate check -> sandbox repro -> severity -> draft fix).
        decision = "stub: pipeline reached worker; agent arrives in Phase 4"
        store.append_step(run_id, "decide", decision)
        store.update(run_id, status="done", decision=decision)
        logger.info("Triage run %s done (stub) for issue #%s", run_id, event.issue_number)
    except Exception as exc:  # noqa: BLE001 — never crash the worker; record and move on.
        logger.exception("Triage run %s failed", run_id)
        try:
            store.append_step(run_id, "error", str(exc))
            store.update(run_id, status="error", decision=f"error: {exc}")
        except Exception:  # noqa: BLE001 — best-effort error recording.
            logger.exception("Could not record error for run %s", run_id)
