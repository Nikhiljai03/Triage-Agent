"""Cross-encoder re-ranking of retrieval candidates.

A bi-encoder (the embedder) is great for cheap recall but mediocre at precise
ordering. We re-score the short-list with a cross-encoder that sees the
(query, candidate) pair jointly, which sharpens the top results. The model is
lazily loaded so importing this module stays offline; tests inject a fake
reranker that implements the same :meth:`rerank` signature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from shared.config import settings

logger = logging.getLogger("triage.rag.rerank")


@dataclass
class Candidate:
    """A retrieval candidate flowing from the vector store into re-ranking."""

    text: str
    issue_id: int
    score: float = 0.0  # vector (bi-encoder) similarity, pre-rerank
    status: str = "open"
    linked_pr: int | None = None
    chunk_type: str = "title_body"
    rerank_score: float | None = None  # cross-encoder score, filled by rerank()
    payload: dict[str, Any] = field(default_factory=dict)


class Reranker:
    """Wraps a sentence-transformers ``CrossEncoder`` (loaded on first use)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading cross-encoder reranker '%s' (first run downloads weights)...",
                self.model_name,
            )
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        """Score each ``(query, candidate.text)`` pair, sort desc, return top_n.

        ``rerank_score`` is set on every candidate (even those beyond ``top_n``)
        so callers can inspect the full ranking if they want.
        """
        if not candidates:
            return []
        model = self._ensure_model()
        scores = model.predict([[query, c.text] for c in candidates])
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)
        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return ranked[:top_n]
