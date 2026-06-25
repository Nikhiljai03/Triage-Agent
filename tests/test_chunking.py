"""Unit tests for token-aware chunking (pure, offline except tiktoken encoding)."""

from __future__ import annotations

import tiktoken

from rag.chunking import chunk_issue
from rag.schemas import IssueComment, IssueRecord

ENC = tiktoken.get_encoding("cl100k_base")


def _make_issue() -> IssueRecord:
    """A synthetic issue with a long body, two comments, and a linked-PR diff."""
    return IssueRecord(
        number=42,
        title="App crashes on startup",
        body="The application crashes immediately on startup. " * 80,  # long -> multi-chunk
        state="closed",
        labels=["bug", "severity:high"],
        comments=[
            IssueComment(author="alice", body="I can reproduce this every time. " * 40),
            IssueComment(author="bob", body="Short confirming comment."),
        ],
        linked_pr=7,
        linked_pr_diff="diff --git a/app.py b/app.py\n" + ("+    fixed_line()\n" * 60),
    )


def test_chunks_respect_max_tokens() -> None:
    chunks = chunk_issue(_make_issue(), max_tokens=64, overlap=8)
    assert chunks
    for chunk in chunks:
        assert len(ENC.encode(chunk.text)) <= 64


def test_all_chunk_types_produced() -> None:
    chunks = chunk_issue(_make_issue(), max_tokens=64, overlap=8)
    types = {c.chunk_type for c in chunks}
    assert types == {"title_body", "comment", "pr_diff"}


def test_long_body_splits_into_multiple_chunks() -> None:
    chunks = chunk_issue(_make_issue(), max_tokens=64, overlap=8)
    title_body_chunks = [c for c in chunks if c.chunk_type == "title_body"]
    assert len(title_body_chunks) > 1  # long body must split


def test_metadata_carried_onto_every_chunk() -> None:
    chunks = chunk_issue(_make_issue(), max_tokens=64, overlap=8)
    for chunk in chunks:
        assert chunk.issue_id == 42
        assert chunk.status == "closed"
        assert chunk.linked_pr == 7
        assert "bug" in chunk.labels
        assert chunk.severity == "high"  # derived from the "severity:high" label


def test_empty_issue_produces_no_chunks() -> None:
    empty = IssueRecord(number=1, title="", body="", state="open")
    assert chunk_issue(empty) == []
