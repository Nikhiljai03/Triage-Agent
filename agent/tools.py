"""The agent's capabilities (plain typed functions the nodes call).

Two groups:

* **Read/compute tools** — ``search_similar_issues`` (Phase 1 RAG),
  ``run_in_sandbox`` (Phase 3 sandbox), ``get_file_contents`` (read a repo file).
* **propose_* tools** — record an *intended* action (label/comment/draft PR) onto
  the state. They DO NOT call GitHub. Actual execution is Phase 5, gated behind
  ``dry_run`` / ``enable_live_writes``. This is the Phase-4 safety contract.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.state import TriageState, log_step
from rag.retrieve import find_similar_issues
from sandbox.runner import SandboxRunner
from sandbox.schemas import ReproRequest, ReproResult
from shared.github_client import GitHubClient
from shared.schemas import IntendedAction

logger = logging.getLogger("triage.agent.tools")

_sandbox_runner: SandboxRunner | None = None
_github_client: GitHubClient | None = None


# --- read / compute tools ---------------------------------------------------
def search_similar_issues(text: str, mode: str, k: int = 5) -> list[dict[str, Any]]:
    """Wrap Phase-1 retrieval; return plain dicts the state/LLM can consume."""
    results = find_similar_issues(text, mode=mode, k=k)  # type: ignore[arg-type]
    return [
        {
            "issue_id": r.issue_id,
            "score": r.score,
            "status": r.status,
            "linked_pr": r.linked_pr,
            "chunk_type": r.chunk_type,
            "text": r.text,
        }
        for r in results
    ]


def run_in_sandbox(
    files: dict[str, str], command: list[str] | str, timeout: int | None = None
) -> ReproResult:
    """Run a snippet in the locked-down Phase-3 sandbox (the only exec path)."""
    global _sandbox_runner
    if _sandbox_runner is None:
        _sandbox_runner = SandboxRunner()
    return _sandbox_runner.run(ReproRequest(files=files, command=command, timeout_seconds=timeout))


def get_file_contents(repo: str, path: str, ref: str | None = None) -> str | None:
    """Read a single file from the repo (read-only) so the fix node can see code."""
    global _github_client
    if _github_client is None:
        _github_client = GitHubClient()
    try:
        repository = _github_client._gh.get_repo(repo)
        contents = repository.get_contents(path, ref=ref) if ref else repository.get_contents(path)
        if isinstance(contents, list):  # a directory, not a file
            return None
        return contents.decoded_content.decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 — best-effort read.
        logger.debug("get_file_contents failed for %s:%s — %s", repo, path, exc)
        return None


# --- propose_* tools (record intent only; NEVER call GitHub) ----------------
def _propose(
    state: TriageState, action_type: str, payload: dict[str, Any], reason: str, *, store: Any = None
) -> None:
    state["intended_actions"].append(
        IntendedAction(type=action_type, payload=payload, reason=reason)  # type: ignore[arg-type]
    )
    summary = ", ".join(f"{k}={v}" for k, v in payload.items() if k != "patch")
    log_step(state, "propose", f"[DRY-RUN] would {action_type} ({summary}) — {reason}", store=store)


def propose_label(state: TriageState, label: str, reason: str, *, store: Any = None) -> None:
    _propose(state, "label", {"label": label}, reason, store=store)


def propose_comment(state: TriageState, body: str, reason: str, *, store: Any = None) -> None:
    _propose(state, "comment", {"body": body}, reason, store=store)


def propose_draft_pr(
    state: TriageState, title: str, body: str, patch: str, reason: str, *, store: Any = None
) -> None:
    _propose(state, "draft_pr", {"title": title, "body": body, "patch": patch}, reason, store=store)
