"""Triage agent graph tests — fully offline (fake LLM, sandbox, and retrieval).

These exercise the real LangGraph state machine and JSON parsing, but inject
canned dependencies so no network, Docker, or model is touched. They also assert
the Phase-4 safety contract: the agent only ever records *proposals*
(``intended_actions``) and never executes a GitHub write (none exist until Phase 5).
"""

from __future__ import annotations

import json

from agent.deps import AgentDeps
from agent.graph import run_triage
from agent.llm import BaseLLM
from sandbox.schemas import ReproResult
from shared.schemas import IssueEvent

_PROPOSAL_TYPES = {"label", "comment", "draft_pr"}


# --- fakes ------------------------------------------------------------------
class FakeLLM(BaseLLM):
    """Routes to a canned JSON response by the schema keyword in the prompt."""

    def __init__(self, *, dup=None, severity=None, fix=None, malformed=False) -> None:
        self._by_key = {"is_duplicate": dup or {}, "can_fix": fix or {}, "severity": severity or {}}
        self._malformed = malformed

    def complete(self, system: str, user: str) -> str:
        if self._malformed:
            return "Sorry — here is some prose instead of JSON."
        low = system.lower()
        for key, response in self._by_key.items():
            if key in low:
                return json.dumps(response)
        return "{}"


class FakeSearch:
    def __init__(self, *, dup=None, fix=None) -> None:
        self.dup, self.fix, self.modes = dup or [], fix or [], []

    def __call__(self, text, mode, k=5):
        self.modes.append(mode)
        return self.dup if mode == "duplicate" else self.fix


class FakeSandbox:
    def __init__(self, result: ReproResult) -> None:
        self.result, self.calls = result, []

    def __call__(self, files, command, timeout=None):
        self.calls.append((files, command, timeout))
        return self.result


def _event(title="App crashes on startup", body="It crashes immediately.", number=10) -> IssueEvent:
    return IssueEvent(
        repo="acme/widgets", issue_number=number, title=title, body=body, action="opened"
    )


def _deps(llm, search, sandbox) -> AgentDeps:
    # store=None -> no run-store mirroring; get_file unused in these paths.
    return AgentDeps(
        llm=llm,
        store=None,
        search_similar=search,
        run_sandbox=sandbox,
        get_file=lambda *a, **k: None,
    )


def _node_names(state) -> list[str]:
    return [s["node"] for s in state["reasoning_log"]]


def _labels(state) -> set[str]:
    return {a.payload.get("label") for a in state["intended_actions"] if a.type == "label"}


def _assert_proposals_only(state) -> None:
    # The whole Phase-4 safety contract: every action is a proposal, never executed.
    assert all(a.type in _PROPOSAL_TYPES for a in state["intended_actions"])


# --- tests ------------------------------------------------------------------
def test_duplicate_path_short_circuits() -> None:
    search = FakeSearch(
        dup=[
            {
                "issue_id": 42,
                "score": 0.95,
                "status": "open",
                "linked_pr": None,
                "chunk_type": "title_body",
                "text": "app crashes on startup with null config",
            }
        ]
    )
    sandbox = FakeSandbox(ReproResult(reproduced=False))
    llm = FakeLLM(
        dup={"is_duplicate": True, "duplicate_of": 42, "confidence": 0.9, "reasoning": "same bug"}
    )

    state = run_triage(_event(), "run-dup", deps=_deps(llm, search, sandbox))

    assert state["is_duplicate"] is True
    assert state["duplicate_of"] == 42
    assert "duplicate" in _labels(state)
    assert any(a.type == "comment" for a in state["intended_actions"])
    # It ended after duplicate_check: no reproduction, severity, or fix work happened.
    assert state["repro_result"] is None
    assert state["severity"] is None
    assert "reproduce" not in _node_names(state)
    assert sandbox.calls == []
    assert "fix" not in search.modes
    _assert_proposals_only(state)


def test_reproduce_and_low_severity_fix_path() -> None:
    search = FakeSearch(
        dup=[],
        fix=[
            {
                "issue_id": 4,
                "score": 0.8,
                "status": "closed",
                "linked_pr": 102,
                "chunk_type": "pr_diff",
                "text": "diff --git a/export.py b/export.py\n+ stream rows",
            }
        ],
    )
    sandbox = FakeSandbox(
        ReproResult(
            reproduced=True, exit_code=1, stdout="", stderr="ValueError: boom", timed_out=False
        )
    )
    llm = FakeLLM(
        dup={
            "is_duplicate": False,
            "duplicate_of": None,
            "confidence": 0.1,
            "reasoning": "different",
        },
        severity={"severity": "low", "confidence": 0.9, "reasoning": "minor edge case"},
        fix={
            "can_fix": True,
            "patch": "diff --git a/x.py b/x.py\n+    fixed()",
            "pr_title": "Fix x",
            "pr_body": "Guards the edge case.",
            "confidence": 0.9,
            "reasoning": "trivial one-liner",
        },
    )
    event = _event(
        title="Crash on parse", body="Repro:\n```python\nraise SystemExit(1)\n```", number=7
    )

    state = run_triage(event, "run-fix", deps=_deps(llm, search, sandbox))

    assert state["is_duplicate"] is False
    assert state["repro_result"] is not None and state["repro_result"]["reproduced"] is True
    assert state["severity"] == "low"
    assert state["drafted_fix"] is not None
    assert state["drafted_fix"]["patch"].startswith("diff --git")
    assert any(a.type == "draft_pr" for a in state["intended_actions"])
    assert "auto-fix-candidate" in _labels(state)
    assert sandbox.calls  # the snippet actually ran in the (fake) sandbox
    assert "fix" in search.modes
    _assert_proposals_only(state)


def test_high_severity_escalates_without_fix() -> None:
    search = FakeSearch(dup=[], fix=[])
    sandbox = FakeSandbox(ReproResult(reproduced=True, exit_code=1))
    llm = FakeLLM(
        dup={"is_duplicate": False, "confidence": 0.1, "reasoning": "no"},
        severity={"severity": "high", "confidence": 0.95, "reasoning": "crash for many users"},
    )
    event = _event(title="Data loss on save", body="```python\nraise SystemExit(1)\n```", number=9)

    state = run_triage(event, "run-esc", deps=_deps(llm, search, sandbox))

    assert state["severity"] == "high"
    assert state["drafted_fix"] is None
    assert "needs-human-review" in _labels(state)
    assert "draft_fix" not in _node_names(state)  # never attempted a draft for high severity
    _assert_proposals_only(state)


def test_malformed_llm_output_escalates_gracefully() -> None:
    search = FakeSearch(
        dup=[
            {
                "issue_id": 1,
                "score": 0.9,
                "status": "open",
                "linked_pr": None,
                "chunk_type": "title_body",
                "text": "something",
            }
        ],
        fix=[],
    )
    sandbox = FakeSandbox(ReproResult(reproduced=False))
    llm = FakeLLM(malformed=True)  # every call returns prose, not JSON

    state = run_triage(
        _event(body="no runnable snippet here", number=3),
        "run-bad",
        deps=_deps(llm, search, sandbox),
    )

    # Defensive handling: no crash; uncertainty routes to escalation.
    assert state["is_duplicate"] is False
    assert state["severity"] is None
    assert state["drafted_fix"] is None
    assert "needs-human-review" in _labels(state)
    assert state["final_decision"]
    _assert_proposals_only(state)
