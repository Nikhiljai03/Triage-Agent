"""Render EVAL_REPORT.md from the dataset + predictions (honest, tight, factual).

Reads ``eval/results/dataset.json`` + ``predictions.json``, computes metrics via
:mod:`eval.metrics` (single source of truth), and writes ``EVAL_REPORT.md`` at the
project root: methodology (with every disclosed shortcut), results tables, a
confusion matrix, an ops/cost table, qualitative traces, and limitations.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.metrics import compute_all

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_REPORT_PATH = Path(__file__).resolve().parent.parent / "EVAL_REPORT.md"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def _by_id(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in predictions}


def _methodology(ds: dict[str, Any]) -> str:
    c = ds["counts"]
    smap = ", ".join(f"`{k}`→`{v}`" for k, v in ds["severity_label_map"].items())
    return f"""## Methodology

All numbers below come from running the **real agent reasoning nodes** in
**dry-run** (no GitHub writes — the execution gate is never invoked, asserted at
startup) over a labeled set built entirely from real GitHub data. Model:
**`{ds['model']}`** (the agent is model-swappable by config; this run used the
model whose free-tier daily quota was available — `gemini-2.5-flash`'s free tier is
capped at 20 requests/day, so the same-family `gemini-2.5-flash-lite` was used).
Retrieval uses the production embedder + cross-encoder reranker
against a controlled Qdrant collection (`{ds['eval_collection']}`) holding
**{len(ds['indexed_issue_ids'])} issues**. Duplicate threshold (model confidence)
= **{ds['confidence_threshold']}**; retrieval similarity threshold =
**{ds['retrieval_similarity_threshold']}**.

**Dataset — {c['total']} cases** ({c['duplicate']} duplicate, {c['distinct']} distinct,
{c['severity']} severity, {c['fix_positive']} fixable, {c['fix_escalate_negative']}
should-escalate).

- **Duplicate / distinct** — source repo `{ds['dup_fix_repo']}`. *Synthetic
  paraphrase method (disclosed):* for a `duplicate` case, an original issue is in
  the index and the input is an LLM-generated paraphrase of it; a correct system
  re-identifies the original. `distinct` negatives are real issues **held out** of
  the index that are not true duplicates of anything indexed (nearest indexed
  neighbour below 0.8 cosine); the **hardest** such negatives — those with the
  highest sub-cap similarity — are chosen, so precision is stress-tested against
  related-but-distinct issues rather than trivially satisfied by an empty
  candidate set.
- **Severity** — source repo `{ds['severity_repo']}`, using its **real
  human-assigned priority labels** mapped 1:1 to our scale: {smap}. No labels were
  invented. Severity is classified from issue text (sandbox repro is disabled for
  this cross-language set).
- **Fix quality** — source repo `{ds['dup_fix_repo']}`. Cases are closed issues
  fixed by a **merged PR**; that PR's diff is the *reference fix*. The agent drafts
  a patch without seeing it, and an LLM judge scores draft-vs-reference for
  same-root-cause (1–5). The fix retriever **excludes the issue's own chunks**
  (leakage guard) so it cannot copy the reference. `should-escalate` negatives are
  `feature request` issues where the correct behaviour is to decline an auto-fix.
"""


def _duplicate_section(m: dict[str, Any]) -> str:
    if not m:
        return ""
    return f"""## Duplicate detection

| Metric | Value |
| --- | --- |
| Precision | {_pct(m['precision'])} |
| Recall | {_pct(m['recall'])} |
| F1 | {_pct(m['f1'])} |
| Accuracy | {_pct(m['accuracy'])} |
| Correct original identified (of TPs) | {_pct(m['correct_identification'])} ({m['correct_identification_count']}/{m['tp']}) |
| Confusion (TP/FP/FN/TN) | {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} (n={m['n']}) |

Positive class = "this issue duplicates one already in the index". A predicted
duplicate counts only when model confidence ≥ the threshold.
"""


def _severity_section(m: dict[str, Any]) -> str:
    if not m:
        return ""
    labels = m["labels"]
    header = "| gold ＼ pred | " + " | ".join(labels) + " | unknown |\n"
    sep = "| --- | " + " | ".join(["---"] * (len(labels) + 1)) + " |\n"
    rows = ""
    for g in labels:
        cells = " | ".join(str(m["confusion_matrix"][g][p]) for p in labels)
        rows += f"| **{g}** | {cells} | {m['confusion_matrix'][g]['unknown']} |\n"
    per = "".join(
        f"| {lbl} | {_pct(v['precision'])} | {_pct(v['recall'])} | {v['support']} |\n"
        for lbl, v in m["per_class"].items()
    )
    return f"""## Severity classification

- **Exact accuracy:** {_pct(m['accuracy'])} ({m['exact_correct']}/{m['n']})
- **Adjacency (±1 rung) accuracy:** {_pct(m['adjacent_accuracy'])} ({m['adjacent_correct']}/{m['n']}) — severity is ordinal, so an off-by-one (e.g. `high` for a gold `critical`) is reported separately, never as exact-correct.

**Confusion matrix** (rows = gold priority label, cols = predicted):

{header}{sep}{rows}
**Per-class** (one-vs-rest):

| class | precision | recall | support |
| --- | --- | --- | --- |
{per}"""


def _fix_section(m: dict[str, Any], model: str) -> str:
    if not m:
        return ""
    dist = " · ".join(f"{s}★×{m['score_distribution'][s]}" for s in range(1, 6))
    return f"""## Fix-draft quality

| Metric | Value |
| --- | --- |
| Cases judged (had a reference fix) | {m['n_judged']} |
| Mean judge score (1–5) | {m['mean_score']} |
| Median judge score | {m['median_score']} |
| Rated reasonable-or-better (≥3) | {_pct(m['pct_reasonable_or_better'])} |
| Addresses root cause (≥4) | {_pct(m['pct_addresses_root_cause'])} |
| `can_fix` decision accuracy (both directions) | {_pct(m['can_fix_accuracy'])} ({m['can_fix_correct']}/{m['can_fix_total']}) |
| Score distribution | {dist} |

Judge = same model (`{model}`) scoring the draft against the merged-PR diff.
Same-model judging is a known bias — see Limitations.
"""


def _ops_section(m: dict[str, Any]) -> str:
    if not m:
        return ""
    rt = "—" if m["total_runtime_s"] is None else f"{m['total_runtime_s']}s"
    return f"""## Operational cost & latency

| Metric | Value |
| --- | --- |
| Mean latency / issue | {m['mean_latency_s']}s |
| Median latency / issue | {m['median_latency_s']}s |
| Mean tokens / issue | {m['mean_tokens_per_issue']:.0f} (in {m['mean_input_tokens_per_issue']:.0f} / out {m['mean_output_tokens_per_issue']:.0f}) |
| Total LLM calls | {m['total_llm_calls']} |
| Total tokens | {m['total_tokens']:,} |
| Actual cost (Gemini free tier) | **$0.00** |
| Est. cost / issue at paid flash rates | ${m['est_cost_per_issue_usd_paid']:.5f} |
| Est. total at paid flash rates | ${m['est_cost_total_usd_paid']:.4f} |
| Total wall-clock (incl. throttle) | {rt} |

Paid-rate estimate uses representative gemini-2.5-flash prices ($0.30 / $2.50 per
1M input / output tokens); throttle delays inflate wall-clock but not token cost.
"""


def _qualitative(ds: dict[str, Any], preds: dict[str, dict[str, Any]]) -> str:
    cases = {c["id"]: c for c in ds["cases"]}

    def trace_block(pid: str) -> str:
        p = preds.get(pid, {})
        lines = "\n".join(f"  - `{s['node']}`: {s['message']}" for s in p.get("trace", []))
        return lines or "  - (no trace)"

    # A win: a duplicate correctly re-identified.
    win = next(
        (
            p
            for p in preds.values()
            if cases.get(p["id"], {}).get("kind") == "duplicate"
            and p.get("prediction", {}).get("is_duplicate")
            and p["prediction"].get("duplicate_of") == cases[p["id"]]["expected"]["duplicate_of"]
        ),
        None,
    )
    # A strong fix.
    best_fix = max(
        (p for p in preds.values() if p.get("judge_score") is not None),
        key=lambda p: p.get("judge_score", 0),
        default=None,
    )
    # A miss: severity wrong, or a fixable case scored low.
    miss = next(
        (
            p
            for p in preds.values()
            if (
                cases.get(p["id"], {}).get("kind") == "severity"
                and p.get("prediction", {}).get("severity")
                != cases[p["id"]]["expected"].get("severity")
            )
            or (p.get("judge_score") is not None and p.get("judge_score", 5) <= 2)
        ),
        None,
    )

    out = ["## Qualitative examples\n"]
    if win:
        c = cases[win["id"]]
        out.append(
            f"**✅ Duplicate re-identified — `{win['id']}`.** Paraphrase of "
            f"pydantic#{c['expected']['duplicate_of']} matched back to it "
            f"(confidence {_num(win['prediction'].get('confidence'))}).\n\n{trace_block(win['id'])}\n"
        )
    if best_fix:
        c = cases[best_fix["id"]]
        out.append(
            f"**✅ Fix draft — `{best_fix['id']}`** (judge {best_fix['judge_score']}/5): "
            f"_{best_fix.get('judge_rationale', '')}_\n\n{trace_block(best_fix['id'])}\n"
        )
    if miss:
        c = cases[miss["id"]]
        if c["kind"] == "severity":
            detail = (
                f"predicted `{miss['prediction'].get('severity')}` vs gold "
                f"`{c['expected'].get('severity')}` (priority label "
                f"`{c['meta'].get('priority_label')}`)"
            )
        else:
            detail = f"judge {miss.get('judge_score')}/5 — _{miss.get('judge_rationale', '')}_"
        out.append(f"**❌ Miss — `{miss['id']}`:** {detail}.\n\n{trace_block(miss['id'])}\n")
    return "\n".join(out)


def _limitations(model: str, predicted: int, total: int) -> str:
    partial = (
        ""
        if predicted >= total
        else (
            f"0. **Partial run.** {predicted}/{total} cases predicted so far; the run "
            f"paused on the Gemini free-tier daily cap (20 requests/day/model) and "
            f"resumes from its checkpoint. Rates below will tighten as the remaining "
            f"cases complete.\n"
        )
    )
    return f"""## Limitations

Disclosed plainly — these bound how far the numbers generalize:

{partial}1. **Small sample.** A few dozen cases; treat every rate as indicative, not
   statistically tight. Per-class severity supports are single digits.
2. **Same-model judging.** Fix quality is scored by the same model that drafts
   (`{model}`); models tend to favour their own style. The judge prompt is
   adversarial and empty drafts are auto-failed, but a human spot-check of a few
   verdicts is the real mitigation.
3. **Synthetic duplicates.** Duplicate positives are LLM paraphrases of real
   issues, not naturally-occurring duplicate reports. This isolates semantic
   matching but is easier than messy real-world dupes.
4. **Multi-repo source.** Duplicate/fix come from `pydantic/pydantic`; severity
   from `rust-lang/rust` (the only one with clean priority labels). Severity
   numbers reflect rust-issue text.
5. **Snippet-only reproduction.** The sandbox reproduce step only fires for issues
   with a runnable python snippet; it was disabled for the (rust) severity set, so
   severity is judged from text alone.
6. **Distinct negatives are constructed.** They are held-out real issues that are
   not true duplicates of anything indexed (nearest neighbour < 0.8 cosine), with
   the hardest mid-similarity cases preferred — a stricter test than empty-retrieval
   negatives, but still a constructed set rather than naturally-labeled duplicates.
"""


def render(ds: dict[str, Any], predictions: list[dict[str, Any]]) -> str:
    metrics = compute_all(ds["cases"], predictions)
    preds = _by_id(predictions)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    head = (
        f"# Evaluation Report — Autonomous Issue-Triage Agent\n\n"
        f"_Generated {today} · model `{ds['model']}` · "
        f"{metrics['counts']['predicted']}/{metrics['counts']['dataset']} cases predicted._\n"
    )
    parts = [
        head,
        _methodology(ds),
        _duplicate_section(metrics["duplicate"]),
        _severity_section(metrics["severity"]),
        _fix_section(metrics["fix"], ds["model"]),
        _ops_section(metrics["ops"]),
        _qualitative(ds, preds),
        _limitations(ds["model"], metrics["counts"]["predicted"], metrics["counts"]["dataset"]),
    ]
    return "\n".join(p for p in parts if p).strip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render EVAL_REPORT.md from results.")
    parser.add_argument("--results-dir", default=str(_RESULTS_DIR))
    parser.add_argument("--out", default=str(_REPORT_PATH))
    args = parser.parse_args(argv)

    results = Path(args.results_dir)
    ds = json.loads((results / "dataset.json").read_text(encoding="utf-8"))
    predictions = json.loads((results / "predictions.json").read_text(encoding="utf-8"))[
        "predictions"
    ]
    Path(args.out).write_text(render(ds, predictions), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
