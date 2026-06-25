"""RunStore round-trip tests against an in-memory fakeredis (fully offline)."""

from __future__ import annotations

import fakeredis

from shared.run_store import RunStore
from shared.schemas import TriageRun


def _store() -> RunStore:
    # decode_responses=True mirrors the real store's string-mode connection.
    return RunStore(connection=fakeredis.FakeRedis(decode_responses=True))


def test_create_get_roundtrip() -> None:
    store = _store()
    store.create(TriageRun(run_id="r1", repo="acme/widgets", issue_number=5))

    got = store.get("r1")
    assert got is not None
    assert got.run_id == "r1"
    assert got.repo == "acme/widgets"
    assert got.issue_number == 5
    assert got.status == "queued"


def test_get_missing_returns_none() -> None:
    assert _store().get("nope") is None


def test_append_step_and_update() -> None:
    store = _store()
    store.create(TriageRun(run_id="r2", repo="acme/widgets", issue_number=7))

    store.append_step("r2", "received", "issue #7")
    store.append_step("r2", "decide", "stub")
    store.update("r2", status="done", decision="stub: done")

    got = store.get("r2")
    assert got is not None
    assert got.status == "done"
    assert got.decision == "stub: done"
    assert [s.name for s in got.steps] == ["received", "decide"]
    assert got.steps[0].detail == "issue #7"
    assert got.updated_at >= got.created_at


def test_list_recent_newest_first_and_capped() -> None:
    store = _store()
    for i in range(3):
        store.create(TriageRun(run_id=f"run{i}", repo="acme/widgets", issue_number=i))

    recent = store.list_recent(limit=2)
    assert [r.run_id for r in recent] == ["run2", "run1"]  # newest first, limited


def test_update_unknown_run_is_noop() -> None:
    assert _store().update("ghost", status="done") is None
