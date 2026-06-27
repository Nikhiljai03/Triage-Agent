"""The guardrail safety matrix — the whole point of Phase 5.

Every test uses a MockGitHubClient and asserts on whether its WRITE methods were
called. No real GitHub calls ever happen here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.executor import execute_intended_actions
from agent.guardrails import execute_action
from shared.config import settings
from shared.schemas import IntendedAction

REPO = "me/scratch"
_WRITE_METHODS = {"add_label", "post_comment", "open_draft_pr"}


class MockGitHubClient:
    """Records every call; lets tests preset idempotency state."""

    def __init__(self, *, labels=None, has_agent_comment=False, branches=None) -> None:
        self.calls: list[tuple] = []
        self._labels = set(labels or [])
        self._has_comment = has_agent_comment
        self._branches = set(branches or [])

    # -- idempotency reads --
    def issue_has_label(self, repo, issue_number, label):
        self.calls.append(("issue_has_label", label))
        return label in self._labels

    def agent_comment_exists(self, repo, issue_number, *a, **k):
        self.calls.append(("agent_comment_exists",))
        return self._has_comment

    def branch_exists(self, repo, branch):
        self.calls.append(("branch_exists", branch))
        return branch in self._branches

    # -- writes --
    def add_label(self, repo, issue_number, label):
        self.calls.append(("add_label", repo, issue_number, label))
        return f"https://github.com/{repo}/issues/{issue_number}"

    def post_comment(self, repo, issue_number, body):
        self.calls.append(("post_comment", repo, issue_number, body))
        return f"https://github.com/{repo}/issues/{issue_number}#issuecomment-1"

    def open_draft_pr(self, repo, issue_number, branch, title, body, patch):
        self.calls.append(("open_draft_pr", repo, issue_number, branch))
        return f"https://github.com/{repo}/pull/1"

    def write_calls(self):
        return [c for c in self.calls if c[0] in _WRITE_METHODS]


def _label():
    return IntendedAction(type="label", payload={"label": "duplicate"}, reason="r")


def _comment():
    return IntendedAction(type="comment", payload={"body": "hello"}, reason="r")


def _draft():
    return IntendedAction(
        type="draft_pr", payload={"title": "T", "body": "B", "patch": "diff"}, reason="r"
    )


def _gates(monkeypatch, dry_run, enable, *, target=REPO, live_repo=""):
    monkeypatch.setattr(settings, "dry_run", dry_run)
    monkeypatch.setattr(settings, "enable_live_writes", enable)
    monkeypatch.setattr(settings, "target_repo", target)
    monkeypatch.setattr(settings, "live_write_repo", live_repo)


@pytest.fixture
def live(monkeypatch):
    """Both gates open, target = REPO, no live_write_repo override."""
    _gates(monkeypatch, dry_run=False, enable=True)


# 1) Default (both gates safe): everything skipped, zero writes.
def test_default_dry_run_skips_all(monkeypatch):
    _gates(monkeypatch, dry_run=True, enable=False)
    client = MockGitHubClient()
    for action in (_label(), _comment(), _draft()):
        assert execute_action(action, REPO, 1, client=client).status == "skipped"
    assert client.write_calls() == []


# 2) Single gate (dry_run=False only): still skipped — proves the SECOND gate is required.
def test_single_gate_dry_run_false_only_skips(monkeypatch):
    _gates(monkeypatch, dry_run=False, enable=False)
    client = MockGitHubClient()
    assert execute_action(_label(), REPO, 1, client=client).status == "skipped"
    assert client.write_calls() == []


# 3) Single gate (enable_live_writes=True only): still skipped — proves the FIRST gate is required.
def test_single_gate_enable_only_skips(monkeypatch):
    _gates(monkeypatch, dry_run=True, enable=True)
    client = MockGitHubClient()
    assert execute_action(_comment(), REPO, 1, client=client).status == "skipped"
    assert client.write_calls() == []


# 4) Both gates open: allowlisted actions execute, with expected write calls.
def test_both_gates_open_executes_allowlisted(live):
    client = MockGitHubClient()
    r_label = execute_action(_label(), REPO, 7, client=client)
    r_comment = execute_action(_comment(), REPO, 7, client=client)
    r_draft = execute_action(_draft(), REPO, 7, client=client)

    assert r_label.status == "executed" and r_label.url
    assert r_comment.status == "executed" and r_comment.url
    assert r_draft.status == "executed" and r_draft.url
    assert {c[0] for c in client.write_calls()} == _WRITE_METHODS


# 5) Allowlist: a non-allowlisted type is refused even with both gates open, no call.
def test_non_allowlisted_refused_even_when_live(live):
    client = MockGitHubClient()
    bad = SimpleNamespace(type="close_issue", payload={}, reason="")
    result = execute_action(bad, REPO, 7, client=client)
    assert result.status == "skipped"
    assert client.write_calls() == []


# 6) Wrong-repo guard: an action for a different repo is refused, no call.
def test_wrong_repo_refused_even_when_live(live):
    client = MockGitHubClient()
    result = execute_action(_comment(), "someone/else", 7, client=client)
    assert result.status == "skipped"
    assert client.write_calls() == []


# 7) Idempotency: existing label / prior agent comment / existing branch -> skipped, no writes.
def test_idempotency_skips_duplicates(live):
    client = MockGitHubClient(
        labels={"duplicate"}, has_agent_comment=True, branches={"triage-agent/issue-7"}
    )
    assert execute_action(_label(), REPO, 7, client=client).status == "skipped"
    assert execute_action(_comment(), REPO, 7, client=client).status == "skipped"
    assert execute_action(_draft(), REPO, 7, client=client).status == "skipped"
    assert client.write_calls() == []


# 8) Write error -> error result; never raises (worker survives).
def test_write_error_becomes_error_result(live):
    class Boom(MockGitHubClient):
        def add_label(self, *a, **k):
            raise RuntimeError("boom")

    client = Boom()
    result = execute_action(_label(), REPO, 7, client=client)
    assert result.status == "error"
    assert "boom" in result.detail
    assert client.write_calls() == []  # the failing write recorded nothing


# Extra: live_write_repo overrides target_repo as the only allowed write target.
def test_live_write_repo_overrides_target(monkeypatch):
    _gates(monkeypatch, dry_run=False, enable=True, target="me/other", live_repo=REPO)
    client = MockGitHubClient()
    assert execute_action(_comment(), REPO, 7, client=client).status == "executed"
    assert execute_action(_comment(), "me/other", 7, client=client).status == "skipped"


# Extra: the executor loops actions, records each, and never writes in dry-run.
def test_executor_dry_run_records_all_skipped(monkeypatch):
    _gates(monkeypatch, dry_run=True, enable=False)
    client = MockGitHubClient()
    state = {
        "event": SimpleNamespace(repo=REPO, issue_number=7),
        "run_id": "r1",
        "intended_actions": [_label(), _comment(), _draft()],
    }
    results = execute_intended_actions(state, client=client, delay=0)
    assert [r.status for r in results] == ["skipped", "skipped", "skipped"]
    assert client.write_calls() == []
