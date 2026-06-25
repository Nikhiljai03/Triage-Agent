"""Webhook signature verification + enqueue behavior (offline: fakeredis + fakes)."""

from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace

import fakeredis
import pytest
from fastapi.testclient import TestClient

import shared.queue as queue_mod
import shared.run_store as run_store_mod
from api.webhook import verify_signature
from shared.config import settings
from shared.run_store import RunStore

SECRET = "topsecret"


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _issue_payload(action: str = "opened", *, is_pr: bool = False) -> dict:
    issue: dict = {
        "number": 99,
        "title": "Crash on startup",
        "body": "It crashes.",
        "html_url": "https://github.com/acme/widgets/issues/99",
        "labels": [{"name": "bug"}],
        "user": {"login": "octocat"},
    }
    if is_pr:
        issue["pull_request"] = {"url": "https://api.github.com/repos/acme/widgets/pulls/99"}
    return {"action": action, "issue": issue, "repository": {"full_name": "acme/widgets"}}


class FakeQueue:
    """Stand-in for an RQ queue: records enqueues, never touches Redis."""

    def __init__(self) -> None:
        self.jobs: list[tuple] = []

    def enqueue(self, func, *args, **kwargs):
        self.jobs.append((func, args, kwargs))
        return SimpleNamespace(id=kwargs.get("job_id", "job"))


# --- verify_signature (unit) -----------------------------------------------
def test_verify_signature_accepts_correct(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_webhook_secret", SECRET)
    body = b'{"a":1}'
    assert verify_signature(body, _sign(body)) is True


def test_verify_signature_rejects_wrong(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_webhook_secret", SECRET)
    assert verify_signature(b'{"a":1}', "sha256=deadbeef") is False


def test_verify_signature_fails_closed_without_secret(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_webhook_secret", "")
    monkeypatch.setattr(settings, "webhook_allow_unsigned", False)
    assert verify_signature(b"anything", None) is False


# --- /webhook (integration via TestClient) ---------------------------------
@pytest.fixture
def client(monkeypatch):
    """A TestClient wired to a fakeredis run store and a fake queue."""
    monkeypatch.setattr(settings, "github_webhook_secret", SECRET)
    store = RunStore(connection=fakeredis.FakeRedis(decode_responses=True))
    fake_queue = FakeQueue()
    # Inject our fakes as the process-wide singletons (auto-restored by monkeypatch).
    monkeypatch.setattr(run_store_mod, "_store_singleton", store)
    monkeypatch.setattr(queue_mod, "_queue_singleton", fake_queue)

    from api.main import app

    return SimpleNamespace(http=TestClient(app), store=store, queue=fake_queue)


def test_signed_opened_returns_202_and_creates_queued_run(client) -> None:
    body = json.dumps(_issue_payload("opened")).encode()
    resp = client.http.post("/webhook", content=body, headers={"X-Hub-Signature-256": _sign(body)})

    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    assert len(client.queue.jobs) == 1  # exactly one job enqueued

    run = client.store.get(run_id)
    assert run is not None
    assert run.status == "queued"
    assert run.issue_number == 99
    assert run.repo == "acme/widgets"


def test_bad_signature_returns_401(client) -> None:
    body = json.dumps(_issue_payload("opened")).encode()
    resp = client.http.post("/webhook", content=body, headers={"X-Hub-Signature-256": "sha256=bad"})
    assert resp.status_code == 401
    assert client.queue.jobs == []  # nothing enqueued on rejection


def test_non_opened_action_is_ignored(client) -> None:
    body = json.dumps(_issue_payload("closed")).encode()
    resp = client.http.post("/webhook", content=body, headers={"X-Hub-Signature-256": _sign(body)})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    assert client.queue.jobs == []


def test_pull_request_payload_is_ignored(client) -> None:
    body = json.dumps(_issue_payload("opened", is_pr=True)).encode()
    resp = client.http.post("/webhook", content=body, headers={"X-Hub-Signature-256": _sign(body)})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    assert client.queue.jobs == []
