"""Central configuration for the Triage Agent.

This module defines a single :class:`Settings` object built on
``pydantic-settings``. Values are loaded from environment variables (or a local
``.env`` file) and exposed through the module-level ``settings`` singleton.

IMPORTANT: nothing else in the codebase should read ``os.environ`` directly.
Always ``from shared.config import settings`` and read attributes off it, so
configuration has exactly one typed source of truth.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed, environment-driven application settings.

    Every variable the whole project will eventually use is declared here with
    a sensible default, so the Phase 0 skeleton boots even without a populated
    ``.env`` (real secrets stay blank by default and are supplied per-deploy).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- GitHub ----------------------------------------------------------
    github_token: str = ""  # PAT / app token to read issues & open PRs
    target_repo: str = "owner/repo"  # "owner/name" of the repository we triage
    github_webhook_secret: str = ""  # shared secret to verify webhook signatures

    # ----- LLM -------------------------------------------------------------
    # The agent reasons via Gemini through its OpenAI-compatible endpoint (so the
    # plain `openai` SDK works). Swapping providers is a config change — point
    # base_url/model/key elsewhere and agent/llm.py needs no edits.
    llm_model: str = "gemini-2.5-flash"  # fast/cheap Gemini model (configurable)
    gemini_api_key: str = ""  # GEMINI_API_KEY — never hardcode
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_api_key: str = ""  # used only by the OpenAI embedding backend (Phase 1)

    # ----- RAG / embeddings ------------------------------------------------
    embedding_backend: str = "sentence-transformers"  # "sentence-transformers" | "openai"
    embedding_model_st: str = "sentence-transformers/all-MiniLM-L6-v2"  # local ST model (dim 384)
    embedding_model_openai: str = "text-embedding-3-small"  # OpenAI embedding model (dim 1536)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # cross-encoder reranker
    qdrant_url: str = "http://qdrant:6333"  # Qdrant REST endpoint
    qdrant_collection: str = "issues"  # collection of issue embeddings
    # On-disk embedding cache (sha256(model+text) -> vector). Relative paths are
    # resolved from the process CWD; first ingestion populates it.
    embedding_cache_dir: str = ".cache/embeddings"
    # Drop retrieval candidates below this cosine similarity in "duplicate" mode.
    retrieval_similarity_threshold: float = 0.5

    # ----- Infra -----------------------------------------------------------
    redis_url: str = "redis://redis:6379/0"  # broker/queue for background jobs

    # ----- Safety ----------------------------------------------------------
    # GitHub writes happen ONLY when BOTH gates are flipped (double opt-in):
    #   dry_run == False  AND  enable_live_writes == True.
    # Defaults below make it impossible to write to a real repo by accident.
    dry_run: bool = True  # if True, never perform live writes
    enable_live_writes: bool = False  # explicit master switch for any GitHub write
    # The ONLY repo writes may target. Empty -> use target_repo. Set this to pin
    # live writes to a specific (e.g. throwaway) repo regardless of the event's repo.
    live_write_repo: str = ""
    confidence_threshold: float = 0.7  # min model confidence before auto-acting
    # Webhook signature verification fails CLOSED: with no secret set, requests are
    # rejected unless this dev-only flag is explicitly enabled.
    webhook_allow_unsigned: bool = False

    # ----- Agent -----------------------------------------------------------
    # Severities (comma-separated) the agent is allowed to attempt an auto-fix for.
    # Conservative default: only "low". Anything else escalates to a human.
    autofix_severities: str = "low"

    # ----- Sandbox ---------------------------------------------------------
    sandbox_image: str = "python:3.11-slim"  # base image for the repro sandbox
    sandbox_timeout_seconds: int = 300  # hard wall-clock limit per sandbox run
    sandbox_mem_limit: str = "512m"  # docker --memory value for the sandbox
    sandbox_cpu_quota: float = 1.0  # fractional CPUs granted to the sandbox


# Module-level singleton — import THIS everywhere, never re-instantiate Settings.
settings = Settings()
