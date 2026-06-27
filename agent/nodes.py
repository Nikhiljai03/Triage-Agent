"""The triage graph nodes — one function per reasoning step.

Each node is ``(state, deps) -> state``: it mutates the shared state and records
*why* via :func:`log_step` (mirrored live to the run store). Decisions are always
derivable from state fields the nodes set, so the routers in ``graph.py`` need no
hidden logic. LLM/JSON failures are caught and handled defensively — an
inconclusive node leaves its fields unset, which the routers send to *escalate*.

Phase-4 safety: nodes only ever call ``propose_*`` (records intent); nothing here
writes to GitHub.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from agent.deps import AgentDeps
from agent.state import TriageState, log_step
from agent.tools import propose_comment, propose_draft_pr, propose_label
from shared.config import settings

logger = logging.getLogger("triage.agent.nodes")

_PROMPT_DIR = Path(__file__).parent / "prompts"
_PY_FENCE = re.compile(r"```(?:python|py)\b\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_VALID_SEVERITIES = {"critical", "high", "medium", "low"}


# --- small helpers ----------------------------------------------------------
def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_json(
    deps: AgentDeps, state: TriageState, node: str, system: str, user: str, schema_hint: str
) -> dict[str, Any] | None:
    """Call the LLM for JSON, turning any failure into a logged ``None`` (no crash)."""
    try:
        return deps.llm.complete_json(system, user, schema_hint)
    except Exception as exc:  # noqa: BLE001 — defensive: bad LLM output -> escalate.
        log_step(state, node, f"LLM/JSON error handled defensively: {exc}", store=deps.store)
        return None


def _extract_python_snippet(text: str | None) -> str | None:
    """Return the first ```python fenced block, if any (pragmatic, not exhaustive)."""
    if not text:
        return None
    match = _PY_FENCE.search(text)
    return match.group(1).strip() if match else None


# --- nodes ------------------------------------------------------------------
def ingest_issue(state: TriageState, deps: AgentDeps) -> TriageState:
    event = state["event"]
    log_step(
        state,
        "ingest_issue",
        f"Triaging {event.repo}#{event.issue_number}: {event.title!r}",
        store=deps.store,
    )
    return state


def retrieve_similar(state: TriageState, deps: AgentDeps) -> TriageState:
    try:
        similar = deps.search_similar(state["issue_text"], "duplicate", k=5)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully if retrieval is down.
        log_step(state, "retrieve_similar", f"retrieval failed: {exc}", store=deps.store)
        similar = []
    state["similar"] = similar
    log_step(
        state, "retrieve_similar", f"found {len(similar)} similar past issue(s)", store=deps.store
    )
    return state


def duplicate_check(state: TriageState, deps: AgentDeps) -> TriageState:
    similar = state.get("similar") or []
    if not similar:
        state["is_duplicate"] = False
        log_step(
            state, "duplicate_check", "no candidates; treating as not a duplicate", store=deps.store
        )
        return state

    lines = [f"NEW ISSUE:\n{state['issue_text']}\n", "CANDIDATE PAST ISSUES:"]
    for i, s in enumerate(similar, 1):
        snippet = (s.get("text") or "")[:300].replace("\n", " ")
        lines.append(
            f"{i}. issue #{s.get('issue_id')} (score {s.get('score') or 0:.2f}): {snippet}"
        )

    result = _safe_json(
        deps,
        state,
        "duplicate_check",
        _load_prompt("duplicate_check.md"),
        "\n".join(lines),
        '{"is_duplicate": bool, "duplicate_of": int|null, "confidence": 0..1, "reasoning": str}',
    )
    if result is None:
        state["is_duplicate"] = False
        log_step(
            state, "duplicate_check", "inconclusive; proceeding as not-duplicate", store=deps.store
        )
        return state

    state["is_duplicate"] = bool(result.get("is_duplicate"))
    state["duplicate_of"] = result.get("duplicate_of")
    state["duplicate_confidence"] = _as_float(result.get("confidence"))
    log_step(
        state,
        "duplicate_check",
        f"is_duplicate={state['is_duplicate']} of #{state['duplicate_of']} "
        f"conf={state['duplicate_confidence']} — {result.get('reasoning', '')}",
        store=deps.store,
    )
    if (
        state["is_duplicate"]
        and (state["duplicate_confidence"] or 0.0) >= settings.confidence_threshold
    ):
        dup = state["duplicate_of"]
        propose_label(state, "duplicate", f"matches #{dup}", store=deps.store)
        propose_comment(
            state,
            # NOTE: the agent never closes issues (Phase-5 guardrails forbid it);
            # this only points to the original — a human decides whether to close.
            f"This appears to be a duplicate of #{dup}. See that issue for the original report.",
            f"link to #{dup}",
            store=deps.store,
        )
        state["final_decision"] = (
            f"Duplicate of #{dup} (confidence {state['duplicate_confidence']})."
        )
    return state


def reproduce(state: TriageState, deps: AgentDeps) -> TriageState:
    snippet = _extract_python_snippet(state["event"].body)
    if not snippet:
        state["repro_result"] = None
        log_step(
            state,
            "reproduce",
            "no runnable python snippet in issue; skipping sandbox",
            store=deps.store,
        )
        return state

    log_step(
        state, "reproduce", "found a python snippet; running it in the sandbox", store=deps.store
    )
    try:
        result = deps.run_sandbox({"repro.py": snippet}, ["python", "repro.py"], 30)
        state["repro_result"] = result.model_dump()
        log_step(
            state,
            "reproduce",
            f"sandbox: reproduced={result.reproduced} exit={result.exit_code} timed_out={result.timed_out}",
            store=deps.store,
        )
    except Exception as exc:  # noqa: BLE001 — sandbox failure must not crash triage.
        state["repro_result"] = None
        log_step(state, "reproduce", f"sandbox error (skipping repro): {exc}", store=deps.store)
    return state


def classify_severity(state: TriageState, deps: AgentDeps) -> TriageState:
    repro = state.get("repro_result")
    parts = [f"ISSUE:\n{state['issue_text']}\n"]
    if repro:
        parts.append(
            f"REPRODUCTION: reproduced={repro.get('reproduced')} exit_code={repro.get('exit_code')} "
            f"timed_out={repro.get('timed_out')}\n"
            f"stdout:\n{(repro.get('stdout') or '')[:1000]}\nstderr:\n{(repro.get('stderr') or '')[:1000]}"
        )
    else:
        parts.append("REPRODUCTION: not attempted (no runnable snippet).")

    result = _safe_json(
        deps,
        state,
        "classify_severity",
        _load_prompt("severity.md"),
        "\n".join(parts),
        '{"severity": "critical|high|medium|low", "confidence": 0..1, "reasoning": str}',
    )
    if result is None:
        log_step(
            state,
            "classify_severity",
            "inconclusive; severity unknown -> will escalate",
            store=deps.store,
        )
        return state

    sev = str(result.get("severity", "")).strip().lower()
    state["severity"] = sev if sev in _VALID_SEVERITIES else None
    state["severity_confidence"] = _as_float(result.get("confidence"))
    state["severity_reasoning"] = result.get("reasoning", "")
    log_step(
        state,
        "classify_severity",
        f"severity={state['severity']} conf={state['severity_confidence']} — {state['severity_reasoning']}",
        store=deps.store,
    )
    return state


def retrieve_fixes(state: TriageState, deps: AgentDeps) -> TriageState:
    try:
        fixes = deps.search_similar(state["issue_text"], "fix", k=3)
    except Exception as exc:  # noqa: BLE001
        log_step(state, "retrieve_fixes", f"fix retrieval failed: {exc}", store=deps.store)
        fixes = []
    state["fix_candidates"] = fixes
    log_step(
        state, "retrieve_fixes", f"found {len(fixes)} past fix(es) with diffs", store=deps.store
    )
    return state


def draft_fix(state: TriageState, deps: AgentDeps) -> TriageState:
    repro = state.get("repro_result")
    parts = [f"ISSUE:\n{state['issue_text']}\n"]
    if repro:
        parts.append(
            f"REPRO: reproduced={repro.get('reproduced')} exit={repro.get('exit_code')}\n"
            f"stderr:\n{(repro.get('stderr') or '')[:800]}"
        )
    parts.append("EXAMPLE FIX DIFFS FROM PAST RESOLVED ISSUES:")
    fixes = state.get("fix_candidates") or []
    if fixes:
        for i, f in enumerate(fixes, 1):
            parts.append(
                f"--- example {i} (issue #{f.get('issue_id')}, PR #{f.get('linked_pr')}):\n"
                f"{(f.get('text') or '')[:1200]}"
            )
    else:
        parts.append("(none found)")

    result = _safe_json(
        deps,
        state,
        "draft_fix",
        _load_prompt("draft_fix.md"),
        "\n".join(parts),
        '{"can_fix": bool, "patch": str, "pr_title": str, "pr_body": str, "confidence": 0..1, "reasoning": str}',
    )
    if result is None:
        log_step(
            state, "draft_fix", "inconclusive; cannot draft a fix -> escalate", store=deps.store
        )
        return state

    can_fix = bool(result.get("can_fix"))
    conf = _as_float(result.get("confidence")) or 0.0
    patch = result.get("patch") or ""
    log_step(
        state,
        "draft_fix",
        f"can_fix={can_fix} conf={conf} — {result.get('reasoning', '')}",
        store=deps.store,
    )

    if can_fix and conf >= settings.confidence_threshold and patch.strip():
        title = result.get("pr_title") or "Automated fix"
        body = result.get("pr_body") or ""
        state["drafted_fix"] = {"patch": patch, "pr_title": title, "pr_body": body}
        propose_draft_pr(
            state, title, body, patch, f"auto-fix candidate (confidence {conf})", store=deps.store
        )
        propose_label(state, "auto-fix-candidate", "agent drafted a minimal fix", store=deps.store)
        state["final_decision"] = (
            f"Drafted a fix PR (confidence {conf}); proposed as auto-fix-candidate. "
            f"No GitHub write performed (dry-run)."
        )
    return state


def escalate(state: TriageState, deps: AgentDeps) -> TriageState:
    reason = _escalation_reason(state)
    propose_label(state, "needs-human-review", reason, store=deps.store)
    if not state.get("final_decision"):
        state["final_decision"] = f"Escalated to human review: {reason}"
    log_step(state, "escalate", state["final_decision"], store=deps.store)
    return state


def _escalation_reason(state: TriageState) -> str:
    sev = state.get("severity")
    if sev is None:
        return "severity could not be determined confidently"
    if sev not in {s.strip().lower() for s in settings.autofix_severities.split(",")}:
        return f"severity '{sev}' is above the auto-fix threshold"
    if state.get("drafted_fix") is None:
        return f"severity '{sev}' but no confident fix could be drafted"
    return "low confidence in the proposed action"
