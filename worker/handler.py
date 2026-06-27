"""Triage job handler — runs the LangGraph agent for one queued issue.

RQ invokes this for each job. It deserializes the :class:`IssueEvent`, marks the
run ``running``, and executes the triage graph. The graph's nodes stream their
reasoning steps into the run store *as they go* (via the default deps' store), so
the dashboard updates live. On completion we record the final decision and a
summary of the agent's *intended* actions — which, per the Phase-4 safety
contract, are proposals only: nothing is written to GitHub.

It must NEVER raise out of the worker process: any failure is caught and recorded
on the run as an ``error`` step + status.
"""

from __future__ import annotations

import logging

from shared.run_store import get_run_store
from shared.schemas import IssueEvent

logger = logging.getLogger("triage.worker.handler")


def handle_triage_job(event_dict: dict, run_id: str) -> None:
    """Process one triage job end to end and persist the outcome."""
    store = get_run_store()
    try:
        event = IssueEvent.model_validate(event_dict)
        store.update(run_id, status="running")

        # Imported here so the worker process starts cheaply and only pays the
        # agent's import cost (langgraph, rag, sandbox) when a job actually runs.
        from agent.graph import run_triage

        final = run_triage(event, run_id)

        actions = final.get("intended_actions") or []
        decision = final.get("final_decision") or "no decision reached"
        summary = ", ".join(
            f"{a.type}({a.payload.get('label', a.payload.get('title', ''))})" for a in actions
        )
        store.append_step(
            run_id,
            "summary",
            f"{len(actions)} intended action(s) [dry-run, not executed]: {summary}",
        )
        store.update(run_id, status="done", decision=decision)
        logger.info("Triage run %s done for issue #%s: %s", run_id, event.issue_number, decision)
    except Exception as exc:  # noqa: BLE001 — never crash the worker; record and move on.
        logger.exception("Triage run %s failed", run_id)
        try:
            store.append_step(run_id, "error", str(exc))
            store.update(run_id, status="error", decision=f"error: {exc}")
        except Exception:  # noqa: BLE001 — best-effort error recording.
            logger.exception("Could not record error for run %s", run_id)
