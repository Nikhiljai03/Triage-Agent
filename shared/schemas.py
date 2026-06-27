"""Cross-service event/run models (shared between the API and the worker).

These are the contracts the event pipeline speaks:

* :class:`GitHubWebhookPayload` — a deliberately loose view of GitHub's large
  ``issues`` webhook body; we only model the fields we read (``extra="ignore"``).
* :class:`IssueEvent` — the normalized issue the rest of the system consumes.
* :class:`TriageRun` (+ :class:`TriageStep`) — the record of one triage attempt,
  the single source of truth the dashboard renders.

(RAG-internal models — ``IssueRecord``/``Chunk`` — live in :mod:`rag.schemas`.)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


# Lifecycle of a triage run as it moves through the pipeline.
RunStatus = Literal["queued", "running", "done", "error"]

# Bug severity buckets the agent classifies into.
Severity = Literal["critical", "high", "medium", "low"]

# The kinds of action the agent may PROPOSE. Phase 5 executes these behind the
# double dry-run gate; the same three (label/comment/draft_pr) are the ONLY types
# ever allowed to write — all non-destructive (no close/merge/delete).
ActionType = Literal["label", "comment", "draft_pr"]

# Outcome of running one intended action through the Phase-5 guardrail gate.
ExecutionStatus = Literal["executed", "skipped", "error"]


class GitHubWebhookPayload(BaseModel):
    """Loose parse of a GitHub ``issues`` webhook event — only the fields we need.

    GitHub payloads are large and evolve; ``extra="ignore"`` keeps this robust.
    ``issue`` / ``repository`` / ``sender`` stay as raw dicts and are normalized
    in :func:`api.webhook.parse_issue_event`.
    """

    model_config = ConfigDict(extra="ignore")

    action: str = ""
    issue: dict[str, Any] | None = None
    repository: dict[str, Any] | None = None
    sender: dict[str, Any] | None = None


class IssueEvent(BaseModel):
    """Normalized issue event handed to the queue and the worker."""

    repo: str
    issue_number: int
    title: str = ""
    body: str = ""
    labels: list[str] = Field(default_factory=list)
    author: str | None = None
    action: str = ""
    html_url: str = ""
    received_at: datetime = Field(default_factory=_utcnow)


class TriageStep(BaseModel):
    """One entry in a run's reasoning trace (richly populated by the agent later)."""

    name: str
    detail: str = ""
    ts: datetime = Field(default_factory=_utcnow)


class IntendedAction(BaseModel):
    """An action the agent decided to take.

    The agent only *records* these (it never writes). Phase 5's guarded executor
    is the sole path that may turn one into a real GitHub write — a label, a
    comment, or a draft PR — and only behind ``dry_run`` / ``enable_live_writes``.
    """

    type: ActionType
    payload: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ExecutionResult(BaseModel):
    """The result of pushing one :class:`IntendedAction` through the guardrail gate.

    ``status`` is ``executed`` (a real GitHub write happened — ``url`` points at it),
    ``skipped`` (gate/allowlist/repo-guard/idempotency declined — the default in
    dry-run), or ``error`` (the write raised; the worker still survives).
    """

    action_type: str
    status: ExecutionStatus
    detail: str = ""
    url: str | None = None


class TriageRun(BaseModel):
    """A single triage attempt and its evolving state/trace."""

    run_id: str
    repo: str
    issue_number: int
    status: RunStatus = "queued"
    steps: list[TriageStep] = Field(default_factory=list)
    decision: str | None = None  # nullable summary, set when the run resolves
    executions: list[ExecutionResult] = Field(default_factory=list)  # Phase-5 write outcomes
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
