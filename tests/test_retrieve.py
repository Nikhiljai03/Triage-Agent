"""Retrieval tests against an in-memory Qdrant with deterministic fakes.

These run fully offline — no Qdrant container, no model downloads — by using
``QdrantClient(":memory:")`` plus a deterministic bag-of-words fake embedder and
a fake reranker. (Run ``docker compose up -d qdrant`` only for real ingestion;
the test suite does not need it.)
"""

from __future__ import annotations

import math

from qdrant_client import QdrantClient

from rag.chunking import chunk_issue
from rag.rerank import Candidate
from rag.retrieve import Retriever
from rag.schemas import IssueRecord
from rag.vector_store import VectorStore

# A tiny fixed vocabulary; each mock issue is "about" a distinct subset of it.
VOCAB = [
    "login",
    "auth",
    "password",
    "crash",
    "startup",
    "memory",
    "leak",
    "export",
    "csv",
    "timeout",
    "button",
    "color",
    "ui",
]


class FakeEmbedder:
    """Deterministic unit-normalized bag-of-words embedder (no models, no I/O)."""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.model_name = "fake-bow"

    @property
    def dim(self) -> int:
        return len(self.vocab)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            low = text.lower()
            vec = [float(low.count(word)) for word in self.vocab]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class FakeReranker:
    """Scores candidates by query-keyword overlap; mirrors the Reranker interface."""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        query_words = [w for w in self.vocab if w in query.lower()]
        for candidate in candidates:
            low = candidate.text.lower()
            candidate.rerank_score = float(sum(low.count(w) for w in query_words))
        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return ranked[:top_n]


def _mock_issues() -> list[IssueRecord]:
    return [
        IssueRecord(
            number=1,
            title="Login fails with wrong password",
            body="Users cannot login, the password auth step rejects valid credentials.",
            state="open",
            labels=["bug"],
        ),
        IssueRecord(
            number=2,
            title="App crash on startup",
            body="The app crash happens on startup before the window appears.",
            state="closed",
            labels=["bug"],
            linked_pr=101,
            linked_pr_diff="diff --git a/startup.py b/startup.py\n+ guard against crash on startup\n",
        ),
        IssueRecord(
            number=3,
            title="Memory leak in worker",
            body="A slow memory leak grows the worker heap over hours.",
            state="open",
            labels=["bug"],
        ),
        IssueRecord(
            number=4,
            title="Export to CSV times out",
            body="The export to csv feature hits a timeout on large datasets.",
            state="closed",
            labels=["bug"],
            linked_pr=102,
            linked_pr_diff="diff --git a/export.py b/export.py\n+ stream csv export to avoid timeout\n",
        ),
        IssueRecord(
            number=5,
            title="Button color wrong in UI",
            body="The primary button color renders incorrectly in the ui theme.",
            state="open",
            labels=["ui"],
        ),
    ]


def _build_retriever() -> Retriever:
    embedder = FakeEmbedder(VOCAB)
    store = VectorStore(client=QdrantClient(":memory:"), collection="test_issues")
    store.ensure_collection(embedder.dim)

    chunks = [c for issue in _mock_issues() for c in chunk_issue(issue)]
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vectors)

    return Retriever(embedder=embedder, store=store, reranker=FakeReranker(VOCAB))


def test_duplicate_mode_returns_matching_issue_first() -> None:
    retriever = _build_retriever()
    results = retriever.find_similar_issues(
        "user cannot login because the password auth is wrong", mode="duplicate", k=3
    )
    assert results
    assert results[0].issue_id == 1  # the login/password issue ranks first
    # De-duplicated by issue: no issue appears twice.
    ids = [r.issue_id for r in results]
    assert len(ids) == len(set(ids))


def test_duplicate_mode_threshold_filters_unrelated() -> None:
    retriever = _build_retriever()
    # A query with no vocabulary overlap should be filtered out by the threshold.
    results = retriever.find_similar_issues(
        "completely unrelated documentation typo", mode="duplicate", k=5, threshold=0.5
    )
    assert results == []


def test_fix_mode_returns_closed_issue_with_diff() -> None:
    retriever = _build_retriever()
    results = retriever.find_similar_issues("export csv timeout on big data", mode="fix", k=3)
    assert results
    top = results[0]
    assert top.issue_id == 4
    assert top.status == "closed"
    assert top.linked_pr == 102
    # Fix mode surfaces the PR diff text for the agent to use as context.
    assert top.chunk_type == "pr_diff"
    assert "csv export" in top.text


def test_fix_mode_excludes_open_issues() -> None:
    retriever = _build_retriever()
    # "memory leak" matches only an OPEN issue (#3) with no linked PR -> excluded.
    results = retriever.find_similar_issues("memory leak in the worker", mode="fix", k=5)
    assert all(r.status == "closed" and r.linked_pr is not None for r in results)
    assert 3 not in [r.issue_id for r in results]


def test_rerank_orders_by_relevance() -> None:
    """The reranker should place the most keyword-relevant issue on top."""
    retriever = _build_retriever()
    results = retriever.find_similar_issues("startup crash", mode="duplicate", k=5)
    assert results[0].issue_id == 2  # crash/startup issue wins the rerank
