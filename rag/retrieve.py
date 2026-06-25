"""High-level retrieval API the agent uses for duplicate detection and fix search.

Two modes:

* ``"duplicate"`` — search across all chunks, drop candidates below a cosine
  threshold, rerank, and return the most similar *distinct* issues. Used to spot
  semantic duplicates of a freshly opened issue.
* ``"fix"`` — restrict to *closed* issues that have a linked PR, pull their
  ``title_body`` + ``pr_diff`` chunks, rerank, and return results whose ``text``
  is the PR diff when available — fix context for a confirmed bug.

Embedder, vector store, and reranker are injectable (defaults are lazy) so the
whole thing runs offline in tests with fakes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from rag.embeddings import Embedder, get_embedder
from rag.rerank import Candidate, Reranker
from rag.vector_store import VectorStore
from shared.config import settings

logger = logging.getLogger("triage.rag.retrieve")

Mode = Literal["duplicate", "fix"]


@dataclass
class RetrievalResult:
    """One retrieved issue, collapsed to its single best chunk."""

    issue_id: int
    score: float
    status: str
    linked_pr: int | None
    text: str
    chunk_type: str


class Retriever:
    """Bundles the embedder + vector store + reranker behind ``find_similar_issues``."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        # Constructors here are cheap/lazy — no model loads or network until used.
        self.embedder = embedder or get_embedder()
        self.store = store or VectorStore()
        self.reranker = reranker or Reranker()

    def find_similar_issues(
        self,
        text: str,
        mode: Mode,
        k: int = 5,
        *,
        candidate_k: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Return up to ``k`` distinct issues most relevant to ``text`` for ``mode``."""
        vector = self.embedder.embed([text])[0]
        candidate_k = candidate_k or max(k * 4, 20)

        if mode == "duplicate":
            thr = settings.retrieval_similarity_threshold if threshold is None else threshold
            points = self.store.search(vector, k=candidate_k, filters=None)
            points = [p for p in points if p.score >= thr]
        elif mode == "fix":
            filters = {"status": "closed", "chunk_type": ["title_body", "pr_diff"]}
            points = self.store.search(vector, k=candidate_k, filters=filters)
            # Keep only issues that actually have a linking PR.
            points = [p for p in points if (p.payload or {}).get("linked_pr") is not None]
        else:  # pragma: no cover — guarded by the Literal type.
            raise ValueError(f"Unknown retrieval mode: {mode!r}")

        candidates = [self._to_candidate(p) for p in points]
        # Rerank the whole short-list, then collapse to one result per issue.
        reranked = self.reranker.rerank(text, candidates, top_n=len(candidates))
        return self._collapse(reranked, mode, k)

    @staticmethod
    def _to_candidate(point) -> Candidate:
        payload = point.payload or {}
        return Candidate(
            text=payload.get("text", ""),
            issue_id=int(payload.get("issue_id")),
            score=float(point.score),
            status=payload.get("status", "open"),
            linked_pr=payload.get("linked_pr"),
            chunk_type=payload.get("chunk_type", "title_body"),
            payload=payload,
        )

    @staticmethod
    def _effective_score(c: Candidate) -> float:
        return c.rerank_score if c.rerank_score is not None else c.score

    def _collapse(self, candidates: list[Candidate], mode: Mode, k: int) -> list[RetrievalResult]:
        """Collapse many chunks per issue into one best result; sort and take ``k``.

        In ``fix`` mode the returned ``text`` prefers the issue's best ``pr_diff``
        chunk (so the agent gets the actual diff), while the ranking ``score``
        still reflects the issue's single best chunk.
        """
        by_issue: dict[int, list[Candidate]] = {}
        for c in candidates:
            by_issue.setdefault(c.issue_id, []).append(c)

        results: list[RetrievalResult] = []
        for issue_chunks in by_issue.values():
            best = max(issue_chunks, key=self._effective_score)
            text_chunk = best
            if mode == "fix":
                diffs = [c for c in issue_chunks if c.chunk_type == "pr_diff"]
                if diffs:
                    text_chunk = max(diffs, key=self._effective_score)
            results.append(
                RetrievalResult(
                    issue_id=best.issue_id,
                    score=self._effective_score(best),
                    status=best.status,
                    linked_pr=best.linked_pr,
                    text=text_chunk.text,
                    chunk_type=text_chunk.chunk_type,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


def find_similar_issues(
    text: str,
    mode: Mode,
    k: int = 5,
    *,
    candidate_k: int | None = None,
    threshold: float | None = None,
) -> list[RetrievalResult]:
    """Module-level convenience that builds a default :class:`Retriever`."""
    return Retriever().find_similar_issues(
        text, mode, k=k, candidate_k=candidate_k, threshold=threshold
    )
