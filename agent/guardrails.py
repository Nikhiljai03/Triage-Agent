"""The ONE and ONLY path permitted to write to GitHub.

Every proposed action flows through :func:`execute_action`, which writes **only**
when ALL of these hold (otherwise it records a ``skipped`` result and returns):

1. ``settings.dry_run is False``      — first gate, AND
2. ``settings.enable_live_writes``    — second, explicit gate (double opt-in), AND
3. ``action.type`` in the :data:`ALLOWLIST` ``{label, comment, draft_pr}`` — all
   non-destructive (no close / merge / delete, ever).

Plus two more refusals even when both gates are open:

* **Target-repo guard** — writes are only ever made to ``live_write_repo`` (or
  ``target_repo`` if unset). A misrouted event targeting any other repo is refused.
* **Idempotency** — before writing we check current state (label present? agent
  already commented? branch already exists?) and skip duplicates.

Defaults (``DRY_RUN=true`` / ``ENABLE_LIVE_WRITES=false``) make it impossible to
write to a real repo by accident — dry-run records what *would* happen, nothing more.
"""

from __future__ import annotations

import logging

from shared.config import settings
from shared.github_client import AGENT_COMMENT_MARKER, GitHubClient
from shared.schemas import ExecutionResult, IntendedAction

logger = logging.getLogger("triage.agent.guardrails")

# The only action types ever permitted to write — all non-destructive.
ALLOWLIST = frozenset({"label", "comment", "draft_pr"})


def allowed_write_repo() -> str:
    """The single repo writes may target (``live_write_repo`` overrides ``target_repo``)."""
    return settings.live_write_repo or settings.target_repo


def _skipped(action_type: str, detail: str) -> ExecutionResult:
    logger.info("guardrail SKIP [%s]: %s", action_type, detail)
    return ExecutionResult(action_type=action_type, status="skipped", detail=detail)


def _executed(action_type: str, detail: str, url: str | None) -> ExecutionResult:
    logger.info("guardrail EXECUTED [%s]: %s -> %s", action_type, detail, url)
    return ExecutionResult(action_type=action_type, status="executed", detail=detail, url=url)


def execute_action(
    action: IntendedAction,
    repo: str,
    issue_number: int,
    *,
    client: GitHubClient | None = None,
) -> ExecutionResult:
    """Run one intended action through the safety gate; write only if all checks pass.

    ``repo`` is the issue's repo (from the event); the guard compares it to the
    single allowed write repo. ``client`` is injectable so tests use a mock and
    never touch GitHub.
    """
    action_type = getattr(action, "type", "unknown")

    # --- GATE 1 + 2: double opt-in (checked FIRST -> default config skips all) --
    if settings.dry_run or not settings.enable_live_writes:
        return _skipped(
            action_type,
            f"dry-run (dry_run={settings.dry_run}, enable_live_writes={settings.enable_live_writes})",
        )

    # --- ALLOWLIST: only non-destructive types, even with both gates open -------
    if action_type not in ALLOWLIST:
        return _skipped(action_type, f"refused: action type '{action_type}' not in allowlist")

    # --- REPO GUARD: only ever write to the allowed repo ------------------------
    allowed = allowed_write_repo()
    if repo != allowed:
        return _skipped(action_type, f"refused: repo '{repo}' != allowed write repo '{allowed}'")

    # Build the real client only now that we've decided we *might* write.
    client = client or GitHubClient()
    payload = getattr(action, "payload", {}) or {}
    try:
        if action_type == "label":
            label = payload.get("label", "")
            if client.issue_has_label(allowed, issue_number, label):
                return _skipped(action_type, f"idempotent: label '{label}' already present")
            return _executed(
                action_type,
                f"added label '{label}'",
                client.add_label(allowed, issue_number, label),
            )

        if action_type == "comment":
            if client.agent_comment_exists(allowed, issue_number):
                return _skipped(action_type, "idempotent: agent already commented on this issue")
            body = f"{payload.get('body', '')}\n\n{AGENT_COMMENT_MARKER}"
            return _executed(
                action_type, "posted comment", client.post_comment(allowed, issue_number, body)
            )

        if action_type == "draft_pr":
            branch = f"triage-agent/issue-{issue_number}"
            if client.branch_exists(allowed, branch):
                return _skipped(action_type, f"idempotent: branch '{branch}' already exists")
            url = client.open_draft_pr(
                allowed,
                issue_number,
                branch,
                payload.get("title") or f"Triage agent: proposed fix for #{issue_number}",
                payload.get("body", ""),
                payload.get("patch", ""),
            )
            return _executed(action_type, "opened draft PR", url)

        return _skipped(action_type, f"refused: unhandled action type '{action_type}'")
    except Exception as exc:  # noqa: BLE001 — a write failure must never crash the worker.
        logger.exception("guardrail ERROR [%s]", action_type)
        return ExecutionResult(action_type=action_type, status="error", detail=str(exc))
