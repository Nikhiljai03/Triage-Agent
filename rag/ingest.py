"""Ingestion pipeline: fetch -> chunk -> embed (cached) -> upsert into Qdrant.

Also the CLI entrypoint::

    python -m rag.ingest --repo OWNER/REPO [--limit N]

Re-running is idempotent: deterministic point IDs mean existing chunks are
updated in place rather than duplicated. Token/config come from
``shared.config.settings`` (set ``GITHUB_TOKEN`` in your environment / ``.env``
for anything beyond GitHub's unauthenticated rate limit).
"""

from __future__ import annotations

import argparse
import logging

from rag.chunking import chunk_issue
from rag.embeddings import Embedder, get_embedder
from rag.vector_store import VectorStore
from shared.github_client import GitHubClient

logger = logging.getLogger("triage.rag.ingest")


def ingest_repo(
    repo: str,
    limit: int | None = None,
    *,
    client: GitHubClient | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    max_tokens: int = 256,
    overlap: int = 32,
) -> int:
    """Run the full pipeline for ``repo``; returns the number of vectors upserted.

    Dependencies are injectable so this can run against an in-memory Qdrant and a
    fake embedder/GitHub client in tests, fully offline.
    """
    client = client or GitHubClient()
    embedder = embedder or get_embedder()
    store = store or VectorStore()

    logger.info("Fetching issues from %s (limit=%s)...", repo, limit)
    issues = client.fetch_issues(repo, state="all", limit=limit)
    logger.info("Fetched %d issues.", len(issues))

    chunks = [
        c for issue in issues for c in chunk_issue(issue, max_tokens=max_tokens, overlap=overlap)
    ]
    logger.info("Created %d chunks.", len(chunks))
    if not chunks:
        logger.warning("No chunks produced; nothing to ingest.")
        return 0

    store.ensure_collection(embedder.dim)
    logger.info("Embedding %d chunks with %s...", len(chunks), embedder.model_name)
    vectors = embedder.embed([c.text for c in chunks])

    upserted = store.upsert(chunks, vectors)
    logger.info("Upserted %d vectors into collection '%s'.", upserted, store.collection)
    return upserted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a GitHub repo's issues into the RAG vector store."
    )
    parser.add_argument("--repo", required=True, help="Target repository as OWNER/REPO.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of issues to ingest.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per chunk.")
    parser.add_argument("--overlap", type=int, default=32, help="Token overlap between chunks.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ingest_repo(args.repo, limit=args.limit, max_tokens=args.max_tokens, overlap=args.overlap)


if __name__ == "__main__":
    main()
