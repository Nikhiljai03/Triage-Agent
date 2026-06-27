"""Run the agent's intended actions through the guardrail gate.

This is the thin loop the worker calls after the graph finishes. It feeds each
:class:`IntendedAction` to :func:`agent.guardrails.execute_action` (the only
permitted write path), collects the typed results, and mirrors each outcome to
the run store so the dashboard shows executed-vs-skipped with reasons and URLs.
A small delay is inserted between *actual* writes to stay rate-limit-friendly.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agent.guardrails import execute_action
from shared.schemas import ExecutionResult

logger = logging.getLogger("triage.agent.executor")

# Polite spacing between real writes (skips/dry-run don't wait).
WRITE_DELAY_SECONDS = 1.0


def execute_intended_actions(
    state: dict[str, Any],
    *,
    client: Any = None,
    store: Any = None,
    delay: float = WRITE_DELAY_SECONDS,
) -> list[ExecutionResult]:
    """Gate-and-run every ``state['intended_actions']``; return one result each."""
    event = state.get("event")
    repo = getattr(event, "repo", "")
    issue_number = getattr(event, "issue_number", 0)
    run_id = state.get("run_id")
    actions = state.get("intended_actions") or []

    results: list[ExecutionResult] = []
    executed_count = 0
    for action in actions:
        if executed_count and delay:
            time.sleep(delay)  # space out real writes only
        result = execute_action(action, repo, issue_number, client=client)
        results.append(result)
        _mirror(store, run_id, result)
        if result.status == "executed":
            executed_count += 1
    return results


def _mirror(store: Any, run_id: str | None, result: ExecutionResult) -> None:
    """Append a human-readable execution step to the run store (best-effort)."""
    verb = {"executed": "EXECUTED", "skipped": "skipped", "error": "ERROR"}[result.status]
    msg = f"[{verb}] {result.action_type}: {result.detail}"
    if result.url:
        msg += f" -> {result.url}"
    logger.info("execute %s", msg)
    if store is not None and run_id:
        try:
            store.append_step(run_id, "execute", msg)
        except Exception:  # noqa: BLE001 — live mirroring is best-effort.
            logger.debug("failed to mirror execution step", exc_info=True)
