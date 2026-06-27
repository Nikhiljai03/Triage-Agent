"""The LangGraph shared state for a single triage run, plus a logging helper.

``TriageState`` is a plain ``TypedDict`` (LangGraph-friendly). Nodes mutate it in
place and return it; the accumulating lists (``reasoning_log``,
``intended_actions``, ...) grow as the graph runs. Every node records *why* it did
what it did via :func:`log_step`, which also mirrors the step to the run store so
the dashboard updates live.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, TypedDict

from shared.schemas import IntendedAction, IssueEvent

logger = logging.getLogger("triage.agent.state")


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class TriageState(TypedDict, total=False):
    """Everything the triage graph reads and writes for one issue."""

    # Input
    event: IssueEvent
    run_id: str
    issue_text: str  # normalized title + body

    # Duplicate detection
    similar: list[dict[str, Any]]
    is_duplicate: bool | None
    duplicate_of: int | None
    duplicate_confidence: float | None

    # Reproduction
    repro_result: dict[str, Any] | None  # serialized ReproResult, or None

    # Severity
    severity: str | None
    severity_confidence: float | None
    severity_reasoning: str | None

    # Fix drafting
    fix_candidates: list[dict[str, Any]]
    drafted_fix: dict[str, Any] | None  # {patch, pr_title, pr_body} | None

    # Outputs (never executed in Phase 4)
    intended_actions: list[IntendedAction]
    reasoning_log: list[dict[str, Any]]
    final_decision: str | None


def init_state(event: IssueEvent, run_id: str) -> TriageState:
    """Build a fresh state with every field initialized to a safe default."""
    issue_text = f"{event.title}\n\n{event.body}".strip()
    return TriageState(
        event=event,
        run_id=run_id,
        issue_text=issue_text,
        similar=[],
        is_duplicate=None,
        duplicate_of=None,
        duplicate_confidence=None,
        repro_result=None,
        severity=None,
        severity_confidence=None,
        severity_reasoning=None,
        fix_candidates=[],
        drafted_fix=None,
        intended_actions=[],
        reasoning_log=[],
        final_decision=None,
    )


def log_step(
    state: TriageState,
    node: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    store: Any = None,
) -> None:
    """Append a reasoning-trace entry and (best-effort) mirror it to the run store.

    Mirroring is wrapped so a run-store/Redis hiccup can never crash the graph;
    in tests ``store`` is ``None`` and only the in-state log is written.
    """
    state["reasoning_log"].append(
        {"node": node, "message": message, "data": data, "ts": _utcnow_iso()}
    )
    logger.info("[%s] %s", node, message)
    if store is not None:
        try:
            store.append_step(state["run_id"], node, message)
        except Exception:  # noqa: BLE001 — live mirroring is best-effort.
            logger.debug("Failed to mirror step to run store", exc_info=True)
