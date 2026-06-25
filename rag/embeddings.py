"""Embedding backends with an on-disk cache.

Two interchangeable implementations sit behind the :class:`Embedder` interface:

* :class:`SentenceTransformerEmbedder` — local, default, dim 384.
* :class:`OpenAIEmbedder` — hosted, dim 1536.

Heavy imports (``sentence_transformers``, ``openai``) are deferred until a model
is actually needed, so importing this module — and running the test suite with a
fake embedder — stays cheap and offline.

Embeddings are cached on disk in a SQLite file keyed by
``sha256(model_name + "\\n" + text)`` so re-ingesting the same issues never
re-computes a vector. The cache location defaults to
``settings.embedding_cache_dir`` (``.cache/embeddings/cache.sqlite3`` relative to
the process CWD) and can be overridden per-embedder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from shared.config import settings

logger = logging.getLogger("triage.rag.embeddings")


def default_cache_path() -> Path:
    """Resolve the SQLite cache file path from settings."""
    base = Path(settings.embedding_cache_dir)
    return base / "cache.sqlite3"


class EmbeddingCache:
    """Tiny SQLite-backed cache mapping ``sha256(model+text)`` -> vector (JSON)."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False keeps it usable from worker threads later.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def make_key(model_name: str, text: str) -> str:
        return hashlib.sha256(f"{model_name}\n{text}".encode()).hexdigest()

    def get(self, key: str) -> list[float] | None:
        row = self._conn.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_many(self, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in items],
        )
        self._conn.commit()


class Embedder(ABC):
    """Embedding interface. Subclasses implement :meth:`_embed_uncached` and :attr:`dim`.

    The shared :meth:`embed` handles cache lookups, batching of the misses, and
    re-assembling results in input order.
    """

    model_name: str

    def __init__(self, cache: EmbeddingCache | None = None) -> None:
        # ``cache=None`` -> use the shared on-disk cache. Tests can pass an
        # in-memory ``EmbeddingCache(":memory:")`` to stay isolated.
        self._cache = cache if cache is not None else EmbeddingCache(default_cache_path())

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` with no cache involvement (subclass responsibility)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, using the cache where possible."""
        if not texts:
            return []

        keys = [EmbeddingCache.make_key(self.model_name, t) for t in texts]
        results: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_text: list[str] = []

        for i, key in enumerate(keys):
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                miss_idx.append(i)
                miss_text.append(texts[i])

        if miss_text:
            vectors = self._embed_uncached(miss_text)
            to_store: list[tuple[str, list[float]]] = []
            for idx, vector in zip(miss_idx, vectors, strict=True):
                results[idx] = vector
                to_store.append((keys[idx], vector))
            self._cache.set_many(to_store)

        return [r for r in results if r is not None]


class SentenceTransformerEmbedder(Embedder):
    """Local embeddings via ``sentence-transformers`` (default backend)."""

    def __init__(
        self,
        model_name: str | None = None,
        cache: EmbeddingCache | None = None,
        batch_size: int = 64,
    ) -> None:
        super().__init__(cache)
        self.model_name = model_name or settings.embedding_model_st
        self._batch_size = batch_size
        self._model = None  # lazy
        self._dim: int | None = None

    def _ensure_model(self):
        if self._model is None:
            # Heavy import + (first run) weight download happen here, not at import.
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading sentence-transformers model '%s' (first run downloads weights)...",
                self.model_name,
            )
            self._model = SentenceTransformer(self.model_name)
            # ST 5.x renamed get_sentence_embedding_dimension -> get_embedding_dimension.
            get_dim = (
                getattr(self._model, "get_embedding_dimension", None)
                or self._model.get_sentence_embedding_dimension
            )
            self._dim = int(get_dim())
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_model()
        assert self._dim is not None
        return self._dim

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


class OpenAIEmbedder(Embedder):
    """Hosted embeddings via the OpenAI API."""

    # Known output dimensions so we never have to make a call just to learn dim.
    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str | None = None,
        cache: EmbeddingCache | None = None,
        batch_size: int = 128,
        api_key: str | None = None,
    ) -> None:
        super().__init__(cache)
        self.model_name = model_name or settings.embedding_model_openai
        self._batch_size = batch_size
        self._api_key = api_key or settings.openai_api_key
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key or None)
        return self._client

    @property
    def dim(self) -> int:
        return self._DIMS.get(self.model_name, 1536)

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = client.embeddings.create(model=self.model_name, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out


def get_embedder(backend: str | None = None, cache: EmbeddingCache | None = None) -> Embedder:
    """Factory selecting an embedder based on ``settings.embedding_backend``."""
    backend = (backend or settings.embedding_backend).lower()
    if backend in ("sentence-transformers", "sentence_transformers", "st", "sbert"):
        return SentenceTransformerEmbedder(cache=cache)
    if backend == "openai":
        return OpenAIEmbedder(cache=cache)
    raise ValueError(f"Unknown embedding_backend: {backend!r}")
