"""LLM-as-judge for fix-draft quality (separate, careful prompt).

Scores the agent's drafted patch against the merged PR diff (the reference fix)
on a 1–5 "same root cause?" rubric. Runs through the SAME metered Gemini model the
agent uses — a deliberate, disclosed limitation (same-model judging has known
self-preference bias; the report flags it and recommends a human spot-check). The
judge call is throttled and metered like every other call, so it counts against
the same quota.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.llm import LLMError
from eval.llm_meter import MeteredGeminiLLM

logger = logging.getLogger("triage.eval.judge")

_PROMPT = (Path(__file__).resolve().parent / "prompts" / "fix_judge.md").read_text(encoding="utf-8")
_SCHEMA_HINT = '{"score": 1-5, "addresses_root_cause": bool, "rationale": str}'


def judge_fix(
    llm: MeteredGeminiLLM,
    issue_text: str,
    reference_diff: str,
    draft_patch: str,
) -> dict[str, object]:
    """Return ``{score:int 1-5, addresses_root_cause:bool, rationale:str}``.

    An empty/absent draft is scored 1 WITHOUT spending an LLM call (nothing to
    judge). Any malformed judge output degrades to a transparent score-1 verdict
    rather than crashing the run.
    """
    if not (draft_patch or "").strip():
        return {"score": 1, "addresses_root_cause": False, "rationale": "no patch was drafted"}

    user = (
        f"ISSUE:\n{issue_text}\n\n"
        f"REFERENCE FIX (merged PR diff):\n{reference_diff}\n\n"
        f"AGENT DRAFT FIX:\n{draft_patch}"
    )
    try:
        result = llm.complete_json(_PROMPT, user, _SCHEMA_HINT)
    except (LLMError, ValueError) as exc:
        logger.warning("judge returned unusable output: %s", exc)
        return {"score": 1, "addresses_root_cause": False, "rationale": f"judge error: {exc}"}

    score = _clamp_score(result.get("score"))
    return {
        "score": score,
        "addresses_root_cause": bool(result.get("addresses_root_cause")) and score >= 4,
        "rationale": str(result.get("rationale", ""))[:300],
    }


def _clamp_score(value: object) -> int:
    try:
        return max(1, min(5, int(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
