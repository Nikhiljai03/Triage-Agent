"""Qdrant vector store: collection management, idempotent upsert, filtered search.

Wraps ``qdrant-client`` against ``settings.qdrant_url`` /
``settings.qdrant_collection``. Point IDs are deterministic (uuid5 of
issue/chunk_type/index) so re-ingesting the same issue *updates* points instead
of creating duplicates. The full chunk text is stored in the payload for
inspection and so retrieval can return it directly.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from rag.schemas import Chunk
from shared.config import settings

logger = logging.getLogger("triage.rag.vector_store")

# Fixed namespace so uuid5 point IDs are stable across processes and runs.
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


class VectorStore:
    """Thin, typed wrapper over a Qdrant collection of issue-chunk vectors."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        url: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.collection = collection or settings.qdrant_collection
        # Injecting a client (e.g. ``QdrantClient(":memory:")``) keeps tests offline.
        self._client = (
            client if client is not None else QdrantClient(url=url or settings.qdrant_url)
        )

    # -- collection management ---------------------------------------------
    def ensure_collection(self, dim: int) -> None:
        """Create the collection (cosine, size ``dim``) if missing.

        If a collection exists with a different vector size, it is recreated —
        the active embedder's dimensionality is the source of truth.
        """
        exists = self._client.collection_exists(self.collection)
        if exists:
            current = self._current_dim()
            if current is not None and current != dim:
                logger.warning(
                    "Collection '%s' has dim %s but embedder dim is %s; recreating.",
                    self.collection,
                    current,
                    dim,
                )
                self._client.delete_collection(self.collection)
                exists = False

        if not exists:
            logger.info("Creating Qdrant collection '%s' (dim=%s, cosine).", self.collection, dim)
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

    def _current_dim(self) -> int | None:
        try:
            info = self._client.get_collection(self.collection)
            return int(info.config.params.vectors.size)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 — version-dependent shape; treat as unknown.
            return None

    # -- writes ------------------------------------------------------------
    def _point_id(self, chunk: Chunk, index: int) -> str:
        raw = f"{chunk.issue_id}:{chunk.chunk_type}:{index}"
        return str(uuid.uuid5(_POINT_NAMESPACE, raw))

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Upsert chunks+vectors. Returns the number of points written.

        ``index`` in the point ID is the position of the chunk *within its
        (issue_id, chunk_type) group*, so IDs are stable regardless of how many
        other issues are ingested in the same batch.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")

        counters: dict[tuple[int, str], int] = {}
        points: list[qm.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            key = (chunk.issue_id, chunk.chunk_type)
            index = counters.get(key, 0)
            counters[key] = index + 1
            payload: dict[str, Any] = {
                "issue_id": chunk.issue_id,
                "status": chunk.status,
                "severity": chunk.severity,
                "labels": chunk.labels,
                "linked_pr": chunk.linked_pr,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,  # raw text kept for inspection / direct return
            }
            points.append(
                qm.PointStruct(id=self._point_id(chunk, index), vector=vector, payload=payload)
            )

        self._client.upsert(collection_name=self.collection, points=points)
        return len(points)

    # -- reads -------------------------------------------------------------
    def search(
        self, vector: list[float], k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[qm.ScoredPoint]:
        """Vector search with optional payload filters (e.g. status, chunk_type)."""
        response = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        return response.points

    def count(self) -> int:
        """Exact number of points currently stored."""
        return self._client.count(collection_name=self.collection, exact=True).count

    def delete_issues(self, issue_ids: list[int]) -> None:
        """Delete every point belonging to the given issue numbers.

        Used by the eval harness to *hold out* chosen distinct-negative issues
        from a freshly built index after selecting them against the full set.
        """
        if not issue_ids:
            return
        self._client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="issue_id", match=qm.MatchAny(any=list(issue_ids)))]
                )
            ),
        )

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> qm.Filter | None:
        """Build a Qdrant ``must`` filter. List values become ``MatchAny`` (OR)."""
        if not filters:
            return None
        must: list[qm.FieldCondition] = []
        for field, value in filters.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                must.append(qm.FieldCondition(key=field, match=qm.MatchAny(any=list(value))))
            else:
                must.append(qm.FieldCondition(key=field, match=qm.MatchValue(value=value)))
        return qm.Filter(must=must) if must else None
