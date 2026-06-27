"""Build the labeled evaluation set from REAL GitHub data — honestly sourced.

Ground truth, per family (all disclosed in EVAL_REPORT.md):

* **duplicate / distinct** — *synthetic-paraphrase* method. An original pydantic
  issue X is indexed; the case input is an LLM paraphrase of X (generated once,
  cached) and the agent should match it back to X (``duplicate``). ``distinct``
  negatives are real pydantic issues HELD OUT of the index whose nearest indexed
  neighbour (excluding themselves) is below the retrieval similarity threshold —
  so a correct system finds no high-confidence match.

* **severity** — REAL human priority labels from ``rust-lang/rust``
  (``P-critical/P-high/P-medium/P-low``) mapped 1:1 to ``critical/high/medium/low``.
  No fabricated labels; the mapping is the identity in all but name.

* **fix** — closed pydantic issues with a MERGED linked PR; the PR diff is the
  reference fix the judge compares the agent's draft against. A few
  ``feature request`` issues are added as ``can_fix=False`` negatives (the agent
  should decline / escalate, not draft a code patch).

The dataset (``eval/results/dataset.json``) and the controlled ``eval_issues``
Qdrant index it populates are stable across runs. Re-running reuses cached GitHub
fetches (``source_issues.json``) and paraphrases (``paraphrases.json``).

CLI::

    python -m eval.dataset --limit 30 [--qdrant-url http://localhost:6333]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from eval.llm_meter import MeteredGeminiLLM, RateLimitStop
from rag.chunking import chunk_issue
from rag.embeddings import Embedder, get_embedder
from rag.schemas import IssueRecord
from rag.vector_store import VectorStore
from shared.config import settings
from shared.github_client import GitHubClient

logger = logging.getLogger("triage.eval.dataset")

EVAL_COLLECTION = "eval_issues"
DUP_FIX_REPO = "pydantic/pydantic"
SEVERITY_REPO = "rust-lang/rust"
# Real priority labels -> our scale. P-critical/high/medium/low is the identity.
SEVERITY_LABEL_MAP: dict[str, str] = {
    "P-critical": "critical",
    "P-high": "high",
    "P-medium": "medium",
    "P-low": "low",
}

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_MAX_INPUT_CHARS = 4000  # bound the text we feed the agent / judge (token sanity)


# --- typed eval case --------------------------------------------------------
class EvalCase(BaseModel):
    """One labeled evaluation case (stable, serialized to dataset.json)."""

    id: str
    kind: Literal["duplicate", "distinct", "severity", "fix"]
    source_repo: str
    issue_id: int
    input_text: str
    expected: dict[str, Any]
    meta: dict[str, Any] = Field(default_factory=dict)


# --- small helpers ----------------------------------------------------------
def _truncate(text: str, n: int = _MAX_INPUT_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "\n…[truncated]"


def _issue_text(rec: IssueRecord) -> str:
    return _truncate(f"{rec.title}\n\n{rec.body}".strip())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _scaled_targets(limit: int | None) -> dict[str, int]:
    """Scale per-family counts to a total ~= ``limit`` (defaults to the full set)."""
    base = {"dup_pos": 8, "distinct": 8, "sev_per_class": 3, "fix": 8, "escalate": 2}
    base_total = (
        base["dup_pos"]
        + base["distinct"]
        + base["sev_per_class"] * 4
        + base["fix"]
        + base["escalate"]
    )
    if not limit or limit >= base_total:
        return base
    scale = limit / base_total
    return {
        "dup_pos": max(2, round(base["dup_pos"] * scale)),
        "distinct": max(2, round(base["distinct"] * scale)),
        "sev_per_class": max(1, round(base["sev_per_class"] * scale)),
        "fix": max(2, round(base["fix"] * scale)),
        "escalate": max(0, round(base["escalate"] * scale)),
    }


# --- GitHub fetch (cached) --------------------------------------------------
def fetch_sources(
    fetch_limit: int,
    sev_per_class: int,
    *,
    client: GitHubClient | None = None,
    refetch: bool = False,
) -> tuple[list[IssueRecord], dict[str, list[IssueRecord]]]:
    """Fetch (and cache) pydantic issues + rust priority-labelled issues."""
    cache = _RESULTS_DIR / "source_issues.json"
    cached = None if refetch else _read_json(cache)
    if cached:
        logger.info("Using cached source issues from %s", cache)
        pyd = [IssueRecord.model_validate(r) for r in cached["pydantic"]]
        sev = {
            lbl: [IssueRecord.model_validate(r) for r in recs]
            for lbl, recs in cached["rust_severity"].items()
        }
        # Top up if the cache is too small for the requested size.
        if len(pyd) >= fetch_limit and all(len(v) >= sev_per_class for v in sev.values()):
            return pyd, sev

    client = client or GitHubClient()
    logger.info(
        "Fetching %d issues from %s (linked PRs, no comments)...", fetch_limit, DUP_FIX_REPO
    )
    pyd = client.fetch_issues(
        DUP_FIX_REPO, state="all", limit=fetch_limit, with_linked_pr=True, with_comments=False
    )
    sev: dict[str, list[IssueRecord]] = {}
    for label in SEVERITY_LABEL_MAP:
        logger.info("Fetching %d %r issues from %s...", sev_per_class * 2, label, SEVERITY_REPO)
        sev[label] = client.fetch_issues(
            SEVERITY_REPO,
            state="all",
            limit=sev_per_class * 2,
            labels=[label],
            with_linked_pr=False,
            with_comments=False,
        )

    _write_json(
        cache,
        {
            "pydantic": [r.model_dump(mode="json") for r in pyd],
            "rust_severity": {
                lbl: [r.model_dump(mode="json") for r in recs] for lbl, recs in sev.items()
            },
        },
    )
    return pyd, sev


# --- controlled eval index --------------------------------------------------
def build_eval_index(
    records: list[IssueRecord], qdrant_url: str, *, embedder: Embedder | None = None
) -> tuple[Embedder, VectorStore]:
    """(Re)build the ``eval_issues`` collection from ``records`` with the real embedder."""
    embedder = embedder or get_embedder()
    store = VectorStore(url=qdrant_url, collection=EVAL_COLLECTION)
    try:
        store._client.delete_collection(EVAL_COLLECTION)  # fresh, deterministic membership
    except Exception:  # noqa: BLE001 — first build: nothing to delete.
        pass
    store.ensure_collection(embedder.dim)
    chunks = [c for r in records for c in chunk_issue(r)]
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vectors)
    logger.info(
        "Built %s with %d chunks from %d issues.", EVAL_COLLECTION, len(chunks), len(records)
    )
    return embedder, store


def _nearest_non_self(store: VectorStore, embedder: Embedder, text: str, issue_id: int) -> float:
    """Best cosine to any indexed issue OTHER than ``issue_id`` (0.0 if none)."""
    vec = embedder.embed([text])[0]
    best = 0.0
    for p in store.search(vec, k=6):
        if (p.payload or {}).get("issue_id") == issue_id:
            continue
        best = max(best, float(p.score))
    return best


# --- paraphrase generation (cached) -----------------------------------------
_PARAPHRASE_SYSTEM = (
    "You rewrite GitHub bug reports. Produce a paraphrase that a DIFFERENT user "
    "might write to report the SAME underlying bug: keep the technical symptom, "
    "the affected component, and any error message identical, but change the "
    "wording, sentence structure, and ordering. Do NOT add new facts, fixes, or "
    "speculation. Return only the rewritten report (a short title line, then body)."
)


def _paraphrase(llm: MeteredGeminiLLM, rec: IssueRecord) -> str:
    user = f"TITLE: {rec.title}\n\nBODY:\n{_truncate(rec.body, 3000)}"
    return _truncate(llm.complete(_PARAPHRASE_SYSTEM, user))


def _load_paraphrases() -> dict[str, str]:
    data = _read_json(_RESULTS_DIR / "paraphrases.json")
    return {str(k): v for k, v in data.items()} if data else {}


# --- assembly ---------------------------------------------------------------
def _interleave(groups: list[list[EvalCase]]) -> list[EvalCase]:
    """Round-robin across kinds so any small prefix (e.g. a 6-case smoke run)
    exercises a mix of duplicate / distinct / severity / fix paths."""
    out: list[EvalCase] = []
    i = 0
    while any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g):
                out.append(g[i])
        i += 1
    return out


def build_dataset(
    limit: int | None,
    qdrant_url: str,
    *,
    delay: float = 0.0,
    refetch: bool = False,
) -> dict[str, Any]:
    targets = _scaled_targets(limit)
    logger.info("Targets: %s", targets)
    pyd, sev_sources = fetch_sources(
        fetch_limit=max(40, targets["dup_pos"] + targets["fix"] + targets["distinct"] + 24),
        sev_per_class=targets["sev_per_class"],
        refetch=refetch,
    )

    # ---- select pydantic issues per family (disjoint) ----
    used: set[int] = set()

    def take(pool: list[IssueRecord], n: int) -> list[IssueRecord]:
        picked = [r for r in pool if r.number not in used][:n]
        used.update(r.number for r in picked)
        return picked

    fix_pool = [
        r
        for r in pyd
        if r.state == "closed"
        and r.linked_pr
        and (r.linked_pr_diff or "").strip()
        and len(r.body) >= 60
        and 80 <= len(r.linked_pr_diff or "") <= 8000
    ]
    fix_recs = take(fix_pool, targets["fix"])

    escalate_pool = [r for r in pyd if "feature request" in r.labels and len(r.body) >= 80]
    escalate_recs = take(escalate_pool, targets["escalate"])

    dup_pool = [r for r in pyd if len(r.title) + len(r.body) >= 150]
    dup_recs = take(dup_pool, targets["dup_pos"])

    # ---- build the full index, then select + hold out distinct negatives ----
    embedder, store = build_eval_index(pyd, qdrant_url)
    distinct_candidates = [r for r in pyd if r.number not in used and len(r.body) >= 80]
    scored = [
        (r, _nearest_non_self(store, embedder, _issue_text(r), r.number))
        for r in distinct_candidates
    ]
    score_by_num = {r.number: s for r, s in scored}
    # A valid distinct negative is a held-out REAL issue that is not a true
    # duplicate of anything indexed. We approximate "not a true duplicate" by
    # excluding candidates whose nearest indexed neighbour is suspiciously high
    # (≥ NON_DUP_CAP — those could genuinely be near-dupes). Among the rest we
    # prefer the HARDEST negatives (highest sub-cap similarity) so the precision
    # number is stress-tested against related-but-distinct issues, not just
    # issues with an empty candidate set.
    non_dup_cap = 0.8
    eligible = sorted(
        [(r, s) for r, s in scored if s < non_dup_cap],
        key=lambda t: t[1],
        reverse=True,
    )
    distinct_recs = [r for r, _ in eligible[: targets["distinct"]]]
    used.update(r.number for r in distinct_recs)
    store.delete_issues([r.number for r in distinct_recs])  # hold them OUT of the index
    indexed_ids = sorted(r.number for r in pyd if r.number not in {x.number for x in distinct_recs})
    logger.info(
        "Index holds %d issues; %d distinct negatives held out.",
        len(indexed_ids),
        len(distinct_recs),
    )

    # ---- paraphrases for duplicate positives (cached + resumable) ----
    para_cache = _load_paraphrases()
    para_path = _RESULTS_DIR / "paraphrases.json"
    llm = MeteredGeminiLLM(delay=delay)
    for r in dup_recs:
        if str(r.number) not in para_cache:
            logger.info("Paraphrasing pydantic#%d ...", r.number)
            para_cache[str(r.number)] = _paraphrase(llm, r)
            _write_json(para_path, para_cache)  # checkpoint each one (resume-safe)

    # ---- assemble cases ----
    dup_cases = [
        EvalCase(
            id=f"dup-{r.number}",
            kind="duplicate",
            source_repo=DUP_FIX_REPO,
            issue_id=r.number,
            input_text=para_cache[str(r.number)],
            expected={"is_duplicate": True, "duplicate_of": r.number},
            meta={"method": "synthetic_paraphrase", "original_title": r.title},
        )
        for r in dup_recs
    ]
    distinct_cases = [
        EvalCase(
            id=f"distinct-{r.number}",
            kind="distinct",
            source_repo=DUP_FIX_REPO,
            issue_id=r.number,
            input_text=_issue_text(r),
            expected={"is_duplicate": False, "duplicate_of": None},
            meta={
                "method": "held_out",
                "nearest_non_self_cosine": round(score_by_num[r.number], 4),
            },
        )
        for r in distinct_recs
    ]
    sev_cases: list[EvalCase] = []
    for label, gold in SEVERITY_LABEL_MAP.items():
        for r in [x for x in sev_sources.get(label, []) if len(x.body) >= 80][
            : targets["sev_per_class"]
        ]:
            sev_cases.append(
                EvalCase(
                    id=f"sev-rust-{r.number}",
                    kind="severity",
                    source_repo=SEVERITY_REPO,
                    issue_id=r.number,
                    input_text=_issue_text(r),
                    expected={"severity": gold},
                    meta={"priority_label": label},
                )
            )
    fix_cases = [
        EvalCase(
            id=f"fix-{r.number}",
            kind="fix",
            source_repo=DUP_FIX_REPO,
            issue_id=r.number,
            input_text=_issue_text(r),
            expected={"can_fix": True},
            meta={
                "linked_pr": r.linked_pr,
                "reference_diff": _truncate(r.linked_pr_diff or "", 8000),
            },
        )
        for r in fix_recs
    ]
    escalate_cases = [
        EvalCase(
            id=f"fix-neg-{r.number}",
            kind="fix",
            source_repo=DUP_FIX_REPO,
            issue_id=r.number,
            input_text=_issue_text(r),
            expected={"can_fix": False},
            meta={"reason": "feature_request_should_escalate", "labels": r.labels},
        )
        for r in escalate_recs
    ]

    cases = _interleave([dup_cases, distinct_cases, sev_cases, fix_cases, escalate_cases])

    dataset = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": settings.llm_model,
        "dup_fix_repo": DUP_FIX_REPO,
        "severity_repo": SEVERITY_REPO,
        "severity_label_map": SEVERITY_LABEL_MAP,
        "eval_collection": EVAL_COLLECTION,
        "qdrant_url": qdrant_url,
        "confidence_threshold": settings.confidence_threshold,
        "retrieval_similarity_threshold": settings.retrieval_similarity_threshold,
        "indexed_issue_ids": indexed_ids,
        "counts": {
            "total": len(cases),
            "duplicate": len(dup_cases),
            "distinct": len(distinct_cases),
            "severity": len(sev_cases),
            "fix_positive": len(fix_cases),
            "fix_escalate_negative": len(escalate_cases),
        },
        "cases": [c.model_dump() for c in cases],
    }
    _write_json(_RESULTS_DIR / "dataset.json", dataset)
    return dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the labeled evaluation dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Approx total number of cases.")
    parser.add_argument(
        "--qdrant-url", default="http://localhost:6333", help="Qdrant for the eval index."
    )
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Seconds between paraphrase LLM calls."
    )
    parser.add_argument("--refetch", action="store_true", help="Ignore the GitHub fetch cache.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        dataset = build_dataset(args.limit, args.qdrant_url, delay=args.delay, refetch=args.refetch)
    except RateLimitStop as stop:
        logger.error("Stopped on rate limit: %s", stop)
        logger.error("Paraphrases generated so far are cached; re-run to resume.")
        raise SystemExit(2) from None
    print(f"Wrote eval/results/dataset.json — {dataset['counts']}")
    print(f"Eval index '{EVAL_COLLECTION}' holds {len(dataset['indexed_issue_ids'])} issues.")


if __name__ == "__main__":
    main()
