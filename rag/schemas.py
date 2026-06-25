"""Typed models for the RAG ingestion + retrieval pipeline.

These are deliberately small, serializable pydantic models so they can move
cleanly between the GitHub fetch layer, the chunker, the vector store payloads,
and the test suite.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# The three kinds of text we embed. Keeping this a Literal lets type checkers and
# Qdrant payload filters agree on the allowed values.
ChunkType = Literal["title_body", "comment", "pr_diff"]


class IssueComment(BaseModel):
    """A single comment on an issue."""

    author: str | None = None
    body: str = ""


class IssueRecord(BaseModel):
    """A raw GitHub issue as fetched by :class:`shared.github_client.GitHubClient`.

    ``linked_pr`` / ``linked_pr_diff`` are populated only when the issue was
    closed by a pull request; otherwise they stay ``None``.
    """

    number: int
    title: str = ""
    body: str = ""
    state: str = "open"  # "open" | "closed"
    labels: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    comments: list[IssueComment] = Field(default_factory=list)
    linked_pr: int | None = None
    linked_pr_diff: str | None = None


class Chunk(BaseModel):
    """A single embeddable unit of text plus the metadata we filter/inspect on.

    ``issue_id`` is the issue *number*. ``severity`` is derived from labels at
    ingest time when a ``severity:*`` / ``priority:*`` label exists, else ``None``.
    """

    text: str
    issue_id: int
    chunk_type: ChunkType
    status: str = "open"
    severity: str | None = None
    labels: list[str] = Field(default_factory=list)
    linked_pr: int | None = None
    created_at: datetime | None = None


def derive_severity(labels: list[str]) -> str | None:
    """Extract a severity/priority value from labels, e.g. ``severity:high`` -> ``"high"``.

    Recognizes ``severity:``/``priority:`` and ``severity/``/``priority/`` styles.
    Returns ``None`` when no such label is present.
    """

    for label in labels:
        low = label.lower()
        if low.startswith(("severity:", "priority:", "severity/", "priority/")):
            for sep in (":", "/"):
                if sep in label:
                    return label.split(sep, 1)[1].strip().lower()
    return None
