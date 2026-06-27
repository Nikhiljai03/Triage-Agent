"""Run the agent over the labeled dataset in DRY-RUN and checkpoint predictions.

Safety: this harness exercises the agent's *reasoning* nodes directly and NEVER
touches the execution gate — ``agent.executor`` / ``agent.guardrails`` are never
imported here, the nodes only ever ``propose_*`` (in-memory, no network), and we
hard-assert ``dry_run`` at startup. So no GitHub write can happen during eval.

Per case we invoke just the relevant nodes:

* duplicate/distinct → ``retrieve_similar`` + ``duplicate_check``
* severity           → ``classify_severity`` (repro disabled by default; the
                        severity corpus is rust, with no python snippet to run)
* fix                → ``retrieve_fixes`` + ``draft_fix`` + LLM judge vs reference

Retrieval is bound to the controlled ``eval_issues`` collection. For FIX cases the
search excludes the issue's OWN chunks (leakage guard) so the agent can't just
copy its own merged-PR diff.

Throttle (``--delay`` between LLM calls, ``--issue-delay`` between cases) and
resume (``--resume`` skips already-checkpointed cases) keep multi-sitting runs
under the Gemini free-tier caps. A persistent 429 (likely the daily cap) raises
``RateLimitStop`` → we checkpoint and exit cleanly; just re-run with ``--resume``.

CLI::

    python -m eval.run_eval [--limit N] [--kinds duplicate,severity,fix]
                            [--delay S] [--issue-delay S] [--resume]
                            [--qdrant-url http://localhost:6333] [--repro]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from agent import nodes
from agent.deps import AgentDeps
from agent.state import init_state
from eval.dataset import EVAL_COLLECTION, EvalCase
from eval.judge import judge_fix
from eval.llm_meter import MeteredGeminiLLM, RateLimitStop
from rag.retrieve import RetrievalResult, Retriever
from rag.vector_store import VectorStore
from shared.config import settings
from shared.schemas import IssueEvent

logger = logging.getLogger("triage.eval.run")

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_DRAFT_LOG_RE = re.compile(r"can_fix=(True|False) conf=([0-9.]+)")


# --- safety -----------------------------------------------------------------
def assert_dry_run() -> None:
    """Refuse to run unless writes are fully disabled (belt-and-suspenders)."""
    if settings.dry_run is not True or settings.enable_live_writes:
        raise SystemExit(
            "REFUSING TO RUN EVAL: writes must be disabled "
            f"(dry_run={settings.dry_run}, enable_live_writes={settings.enable_live_writes}). "
            "Set DRY_RUN=true and ENABLE_LIVE_WRITES=false."
        )


# --- retrieval bound to the eval index (with fix-mode leakage guard) --------
def _result_to_dict(r: RetrievalResult) -> dict[str, Any]:
    return {
        "issue_id": r.issue_id,
        "score": r.score,
        "status": r.status,
        "linked_pr": r.linked_pr,
        "chunk_type": r.chunk_type,
        "text": r.text,
    }


def make_search(retriever: Retriever, exclude_issue_id: int | None):
    """A ``search_similar(text, mode, k)`` bound to the eval index.

    ``exclude_issue_id`` (set for fix cases) drops the issue's own chunks so the
    agent cannot retrieve — and copy — its own reference fix.
    """

    def search(text: str, mode: str, k: int = 5) -> list[dict[str, Any]]:
        want = k + (1 if exclude_issue_id is not None else 0)
        results = retriever.find_similar_issues(text, mode=mode, k=want)  # type: ignore[arg-type]
        rows = [_result_to_dict(r) for r in results if r.issue_id != exclude_issue_id]
        return rows[:k]

    return search


# --- checkpoint IO ----------------------------------------------------------
def _load_predictions(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"meta": {}, "predictions": []}


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


# --- per-case execution -----------------------------------------------------
def _event_for(case: EvalCase) -> IssueEvent:
    # issue_number is set to 0 for paraphrase cases so a duplicate self-match is by
    # content, never by number; real number is kept in case.issue_id for scoring.
    return IssueEvent(repo=case.source_repo, issue_number=0, title="", body=case.input_text)


def _trace(state: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"node": s["node"], "message": str(s["message"])[:300]}
        for s in state.get("reasoning_log", [])
    ]


def run_case(
    case: EvalCase, retriever: Retriever, llm: MeteredGeminiLLM, *, repro: bool
) -> dict[str, Any]:
    exclude = case.issue_id if case.kind == "fix" else None
    deps = AgentDeps(llm=llm, store=None, search_similar=make_search(retriever, exclude))
    state = init_state(_event_for(case), run_id=f"eval-{case.id}")

    before = llm.usage_snapshot()
    t0 = time.monotonic()
    prediction: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    if case.kind in ("duplicate", "distinct"):
        nodes.retrieve_similar(state, deps)
        nodes.duplicate_check(state, deps)
        prediction = {
            "is_duplicate": state.get("is_duplicate"),
            "duplicate_of": state.get("duplicate_of"),
            "confidence": state.get("duplicate_confidence"),
        }
    elif case.kind == "severity":
        if repro:
            nodes.reproduce(state, deps)
        nodes.classify_severity(state, deps)
        prediction = {
            "severity": state.get("severity"),
            "confidence": state.get("severity_confidence"),
        }
    elif case.kind == "fix":
        nodes.retrieve_fixes(state, deps)
        nodes.draft_fix(state, deps)
        drafted = state.get("drafted_fix")
        raw = _parse_draft_log(state)
        prediction = {
            "can_fix": bool(drafted),
            "confidence": raw.get("confidence"),
            "pr_title": (drafted or {}).get("pr_title", ""),
            "patch_excerpt": (drafted or {}).get("patch", "")[:1500],
        }
        extra["raw_can_fix"] = raw.get("can_fix")
        if case.expected.get("can_fix") and case.meta.get("reference_diff"):
            verdict = judge_fix(
                llm, case.input_text, case.meta["reference_diff"], (drafted or {}).get("patch", "")
            )
            extra["judge_score"] = verdict["score"]
            extra["addresses_root_cause"] = verdict["addresses_root_cause"]
            extra["judge_rationale"] = verdict["rationale"]

    latency = time.monotonic() - t0
    usage = MeteredGeminiLLM.diff(before, llm.usage_snapshot())
    record = {
        "id": case.id,
        "kind": case.kind,
        "issue_id": case.issue_id,
        "latency_s": round(latency, 3),
        "llm_calls": int(usage["calls"]),
        "input_tokens": int(usage["input_tokens"]),
        "output_tokens": int(usage["output_tokens"]),
        "total_tokens": int(usage["total_tokens"]),
        "prediction": prediction,
        "trace": _trace(state),
        **extra,
    }
    return record


def _parse_draft_log(state: dict[str, Any]) -> dict[str, Any]:
    """Best-effort: recover the model's raw can_fix/confidence from the trace."""
    for step in reversed(state.get("reasoning_log", [])):
        if step.get("node") == "draft_fix":
            m = _DRAFT_LOG_RE.search(str(step.get("message", "")))
            if m:
                return {"can_fix": m.group(1) == "True", "confidence": float(m.group(2))}
    return {}


# --- driver -----------------------------------------------------------------
def _family(kind: str) -> str:
    return "duplicate" if kind in ("duplicate", "distinct") else kind


def run_eval(
    *,
    limit: int | None,
    kinds: list[str] | None,
    delay: float,
    issue_delay: float,
    resume: bool,
    qdrant_url: str,
    repro: bool,
) -> dict[str, Any]:
    assert_dry_run()
    dataset = json.loads((_RESULTS_DIR / "dataset.json").read_text(encoding="utf-8"))
    cases = [EvalCase.model_validate(c) for c in dataset["cases"]]
    if kinds:
        wanted = set(kinds)
        cases = [c for c in cases if _family(c.kind) in wanted]

    pred_path = _RESULTS_DIR / "predictions.json"
    store = _load_predictions(pred_path) if resume else {"meta": {}, "predictions": []}
    done = {p["id"] for p in store["predictions"]}
    store["meta"] = {
        "model": settings.llm_model,
        "eval_collection": EVAL_COLLECTION,
        "qdrant_url": qdrant_url,
        "delay": delay,
        "issue_delay": issue_delay,
        "repro": repro,
        "dry_run": settings.dry_run,
        "enable_live_writes": settings.enable_live_writes,
    }

    todo = [c for c in cases if c.id not in done]
    if limit is not None:
        todo = todo[:limit]
    logger.info(
        "Eval: %d case(s) to run (%d already done, %d in dataset). dry_run=%s",
        len(todo),
        len(done),
        len(cases),
        settings.dry_run,
    )

    llm = MeteredGeminiLLM(delay=delay)
    retriever = Retriever(store=VectorStore(url=qdrant_url, collection=EVAL_COLLECTION))

    run_started = time.monotonic()
    for i, case in enumerate(todo, 1):
        logger.info("[%d/%d] %s (%s)", i, len(todo), case.id, case.kind)
        try:
            record = run_case(case, retriever, llm, repro=repro)
        except RateLimitStop as stop:
            logger.error("Rate limit reached: %s", stop)
            logger.error(
                "Checkpointed %d prediction(s); re-run with --resume to continue.", len(done)
            )
            store["meta"]["stopped_on_rate_limit"] = str(stop)
            _atomic_write(pred_path, store)
            raise SystemExit(2) from None
        except Exception as exc:  # noqa: BLE001 — record per-case failure, keep going.
            logger.exception("case %s failed", case.id)
            record = {
                "id": case.id,
                "kind": case.kind,
                "issue_id": case.issue_id,
                "error": str(exc),
            }

        store["predictions"].append(record)
        done.add(case.id)
        store["meta"]["total_runtime_s"] = round(time.monotonic() - run_started, 1)
        _atomic_write(pred_path, store)  # checkpoint after EVERY case
        if issue_delay and i < len(todo):
            time.sleep(issue_delay)

    print(f"Done. {len(store['predictions'])} prediction(s) in {pred_path}")
    return store


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the agent over the eval dataset (dry-run).")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max NEW cases to run this sitting."
    )
    parser.add_argument("--kinds", default=None, help="Comma list: duplicate,severity,fix.")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between LLM calls.")
    parser.add_argument("--issue-delay", type=float, default=2.0, help="Seconds between cases.")
    parser.add_argument("--resume", action="store_true", help="Skip already-checkpointed cases.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument(
        "--repro", action="store_true", help="Run the sandbox repro step for severity."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    run_eval(
        limit=args.limit,
        kinds=kinds,
        delay=args.delay,
        issue_delay=args.issue_delay,
        resume=args.resume,
        qdrant_url=args.qdrant_url,
        repro=args.repro,
    )


if __name__ == "__main__":
    main()
