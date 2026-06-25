"""Token-aware chunking of issues into embeddable :class:`Chunk` objects.

Pure functions only — no network, no models beyond the tiktoken encoding — so
this module is trivially unit-testable. We chunk by token budget (not characters)
so chunks line up with what the embedding model actually consumes.
"""

from __future__ import annotations

import tiktoken

from rag.schemas import Chunk, IssueRecord, derive_severity

# cl100k_base is the encoding used by OpenAI text-embedding-3-* and is a fine,
# widely-compatible proxy for token counting with sentence-transformers too.
_ENCODING_NAME = "cl100k_base"
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Lazily load and cache the tiktoken encoding (first call may hit the network)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoder


def _split_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split ``text`` into pieces of at most ``max_tokens`` tokens, with overlap.

    Returns ``[]`` for empty/whitespace-only input. Each returned piece is
    guaranteed to encode to ``<= max_tokens`` tokens.
    """

    text = text.strip()
    if not text:
        return []

    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    n = len(tokens)
    pieces: list[str] = []
    start = 0
    while start < n:
        target_end = min(start + max_tokens, n)
        # tiktoken decode(encode(...)) is not idempotent at arbitrary token
        # boundaries, so a decoded window can *re-encode* to slightly more than
        # max_tokens. Shrink the window until the re-encoded piece fits the budget.
        end = target_end
        piece = enc.decode(tokens[start:end]).strip()
        while end > start + 1 and len(enc.encode(piece)) > max_tokens:
            end -= 1
            piece = enc.decode(tokens[start:end]).strip()
        if piece:
            pieces.append(piece)
        if target_end >= n:
            break
        start += step
    return pieces


def chunk_issue(issue: IssueRecord, max_tokens: int = 256, overlap: int = 32) -> list[Chunk]:
    """Chunk an issue into title+body, per-comment, and linked-PR-diff chunks.

    * ``title_body`` — the title and body chunked together.
    * ``comment``    — each comment chunked separately (long ones split with overlap).
    * ``pr_diff``    — the linked PR's diff, if present.

    Metadata (issue id, status, derived severity, labels, linked PR, created_at)
    is carried onto every chunk. No chunk ever exceeds ``max_tokens`` tokens.
    """

    common = {
        "issue_id": issue.number,
        "status": issue.state,
        "severity": derive_severity(issue.labels),
        "labels": list(issue.labels),
        "linked_pr": issue.linked_pr,
        "created_at": issue.created_at,
    }
    chunks: list[Chunk] = []

    title_body = f"{issue.title}\n\n{issue.body}".strip()
    for piece in _split_tokens(title_body, max_tokens, overlap):
        chunks.append(Chunk(text=piece, chunk_type="title_body", **common))

    for comment in issue.comments:
        for piece in _split_tokens(comment.body, max_tokens, overlap):
            chunks.append(Chunk(text=piece, chunk_type="comment", **common))

    if issue.linked_pr_diff:
        for piece in _split_tokens(issue.linked_pr_diff, max_tokens, overlap):
            chunks.append(Chunk(text=piece, chunk_type="pr_diff", **common))

    return chunks
