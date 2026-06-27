"""Metric math for the evaluation harness — pure functions, no network, no deps.

Everything here is a plain function over lists of dicts (predictions joined with
ground truth), so it is trivially unit-testable offline (see
``tests/test_metrics.py``). We compute precision/recall/F1/accuracy and the
confusion matrix BY HAND rather than pulling in scikit-learn: the math is small,
keeping it dependency-free means the metric tests run with zero install and the
numbers are auditable line-by-line.

The module also exposes a tiny CLI (``python -m eval.metrics``) that joins
``eval/results/dataset.json`` with ``eval/results/predictions.json`` and writes
``eval/results/metrics.json`` — but that IO lives in :func:`main`; the metric
functions themselves stay pure.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# Severity scale, ordered most→least severe. The index is used for the ordinal
# "off-by-one" (adjacency) tolerance, since severity is ordinal, not nominal.
SEVERITY_LEVELS: list[str] = ["critical", "high", "medium", "low"]

# Representative PAID per-1M-token prices for gemini-2.5-flash, used ONLY to
# translate measured token usage into an illustrative dollar figure. The eval
# itself runs on the free tier ($0). These are approximate and configurable.
FLASH_INPUT_PRICE_PER_1M = 0.30
FLASH_OUTPUT_PRICE_PER_1M = 2.50

_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --- primitives -------------------------------------------------------------
def safe_div(num: float, den: float) -> float:
    """Divide, returning 0.0 on a zero denominator (so empty slices don't crash)."""
    return num / den if den else 0.0


def binary_prf(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Precision/recall/F1/accuracy for a binary confusion (counts in, rates out)."""
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# --- duplicate detection ----------------------------------------------------
def duplicate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision/recall/F1/accuracy for duplicate vs. distinct cases.

    Each row needs ``gold_is_duplicate`` / ``pred_is_duplicate`` (bool). Positive
    class = "this issue IS a duplicate of something in the index". We also report
    ``correct_identification``: of the true-positive matches, the fraction where
    the predicted ``duplicate_of`` issue number equals the gold one.
    """
    tp = fp = fn = tn = 0
    correct_id = 0
    for r in rows:
        gold = bool(r.get("gold_is_duplicate"))
        pred = bool(r.get("pred_is_duplicate"))
        if gold and pred:
            tp += 1
            if r.get("pred_duplicate_of") == r.get("gold_duplicate_of"):
                correct_id += 1
        elif not gold and pred:
            fp += 1
        elif gold and not pred:
            fn += 1
        else:
            tn += 1
    out = binary_prf(tp, fp, fn, tn)
    out["n"] = len(rows)
    out["correct_identification"] = safe_div(correct_id, tp)
    out["correct_identification_count"] = correct_id
    return out


# --- severity classification ------------------------------------------------
def confusion_matrix(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, dict[str, int]]:
    """Nominal confusion matrix gold→pred. A pred not in ``labels`` (e.g. ``None``
    when the model was inconclusive) is bucketed under the ``"unknown"`` column."""
    columns = list(labels) + ["unknown"]
    matrix = {g: {p: 0 for p in columns} for g in labels}
    for r in rows:
        gold = r.get("gold")
        pred = r.get("pred")
        if gold not in matrix:
            continue
        col = pred if pred in labels else "unknown"
        matrix[gold][col] += 1
    return matrix


def severity_metrics(rows: list[dict[str, Any]], labels: list[str] | None = None) -> dict[str, Any]:
    """Exact + adjacency accuracy, confusion matrix, and per-class P/R for severity.

    Each row needs ``gold`` and ``pred`` (a severity string, or ``None``/unknown
    pred). ``adjacent_accuracy`` counts a prediction correct when it is exactly
    right OR one rung off on the ordinal scale (e.g. predicted ``high`` for a gold
    ``critical``) — reported alongside, never instead of, exact accuracy.
    """
    labels = labels or SEVERITY_LEVELS
    index = {lbl: i for i, lbl in enumerate(labels)}
    n = len(rows)
    exact = 0
    adjacent = 0
    for r in rows:
        gold, pred = r.get("gold"), r.get("pred")
        if pred == gold:
            exact += 1
            adjacent += 1
        elif pred in index and gold in index and abs(index[pred] - index[gold]) <= 1:
            adjacent += 1

    matrix = confusion_matrix(rows, labels)
    per_class: dict[str, dict[str, float]] = {}
    for lbl in labels:
        tp = sum(1 for r in rows if r.get("gold") == lbl and r.get("pred") == lbl)
        fp = sum(1 for r in rows if r.get("gold") != lbl and r.get("pred") == lbl)
        fn = sum(1 for r in rows if r.get("gold") == lbl and r.get("pred") != lbl)
        per_class[lbl] = {
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "support": sum(1 for r in rows if r.get("gold") == lbl),
        }
    return {
        "n": n,
        "accuracy": safe_div(exact, n),
        "adjacent_accuracy": safe_div(adjacent, n),
        "exact_correct": exact,
        "adjacent_correct": adjacent,
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


# --- fix quality ------------------------------------------------------------
def fix_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate LLM-judge scores + can_fix correctness for the fix family.

    Rows may mix *fixable* cases (have a reference diff and a judge score) and
    *should-escalate* negatives (``gold_can_fix=False``, no judge score). Judge
    aggregates are over rows that actually carry a ``judge_score``; ``can_fix``
    accuracy is over every fix-family row (both directions).
    """
    judged = [r for r in rows if r.get("judge_score") is not None]
    scores = [int(r["judge_score"]) for r in judged]
    reasonable = [s for s in scores if s >= 3]
    addresses = [r for r in judged if r.get("addresses_root_cause")]

    can_fix_rows = [r for r in rows if "gold_can_fix" in r and "pred_can_fix" in r]
    can_fix_correct = sum(
        1 for r in can_fix_rows if bool(r["pred_can_fix"]) == bool(r["gold_can_fix"])
    )
    return {
        "n": len(rows),
        "n_judged": len(judged),
        "mean_score": round(statistics.fmean(scores), 3) if scores else 0.0,
        "median_score": statistics.median(scores) if scores else 0.0,
        "pct_reasonable_or_better": safe_div(len(reasonable), len(judged)),
        "pct_addresses_root_cause": safe_div(len(addresses), len(judged)),
        "can_fix_accuracy": safe_div(can_fix_correct, len(can_fix_rows)),
        "can_fix_correct": can_fix_correct,
        "can_fix_total": len(can_fix_rows),
        "score_distribution": {s: scores.count(s) for s in range(1, 6)},
    }


# --- ops / cost / latency ---------------------------------------------------
def ops_metrics(
    rows: list[dict[str, Any]],
    *,
    total_runtime_s: float | None = None,
    input_price_per_1m: float = FLASH_INPUT_PRICE_PER_1M,
    output_price_per_1m: float = FLASH_OUTPUT_PRICE_PER_1M,
) -> dict[str, Any]:
    """Latency / token / cost aggregates over per-case ops records.

    Each row may carry ``latency_s``, ``input_tokens``, ``output_tokens``,
    ``total_tokens`` and ``llm_calls``. Token/cost aggregates ignore rows with no
    usage data (e.g. a case skipped before any LLM call). Cost is an *estimate*
    at representative paid flash rates; the run itself is free-tier ($0).
    """
    latencies = [float(r["latency_s"]) for r in rows if r.get("latency_s") is not None]
    tok_rows = [r for r in rows if r.get("total_tokens") is not None]
    in_tok = sum(int(r.get("input_tokens") or 0) for r in tok_rows)
    out_tok = sum(int(r.get("output_tokens") or 0) for r in tok_rows)
    total_tok = sum(int(r.get("total_tokens") or 0) for r in tok_rows)
    calls = sum(int(r.get("llm_calls") or 0) for r in rows)
    n_tok = len(tok_rows) or 1

    est_cost_total = (in_tok / 1_000_000) * input_price_per_1m + (
        out_tok / 1_000_000
    ) * output_price_per_1m
    return {
        "n": len(rows),
        "mean_latency_s": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "median_latency_s": round(statistics.median(latencies), 3) if latencies else 0.0,
        "total_llm_calls": calls,
        "mean_tokens_per_issue": round(total_tok / n_tok, 1),
        "mean_input_tokens_per_issue": round(in_tok / n_tok, 1),
        "mean_output_tokens_per_issue": round(out_tok / n_tok, 1),
        "total_tokens": total_tok,
        "est_cost_total_usd_paid": round(est_cost_total, 4),
        "est_cost_per_issue_usd_paid": round(safe_div(est_cost_total, len(rows)), 5),
        "actual_cost_usd": 0.0,  # free tier
        "total_runtime_s": round(total_runtime_s, 1) if total_runtime_s is not None else None,
    }


# --- join + CLI (IO lives here; the functions above stay pure) --------------
def _index_by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(it["id"]): it for it in items}


def compute_all(dataset: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Join gold (dataset) with predictions by case id and compute every family."""
    preds = _index_by_id(predictions)

    dup_rows: list[dict[str, Any]] = []
    sev_rows: list[dict[str, Any]] = []
    fix_rows: list[dict[str, Any]] = []
    ops_rows: list[dict[str, Any]] = []

    for case in dataset:
        cid = str(case["id"])
        pred = preds.get(cid)
        if pred is None:
            continue  # not yet predicted (partial run) — skip, don't fabricate
        kind = case["kind"]
        expected = case.get("expected", {})
        p = pred.get("prediction", {})
        ops_rows.append(
            {
                "latency_s": pred.get("latency_s"),
                "input_tokens": pred.get("input_tokens"),
                "output_tokens": pred.get("output_tokens"),
                "total_tokens": pred.get("total_tokens"),
                "llm_calls": pred.get("llm_calls"),
            }
        )
        if kind in ("duplicate", "distinct"):
            dup_rows.append(
                {
                    "gold_is_duplicate": bool(expected.get("is_duplicate")),
                    "pred_is_duplicate": bool(p.get("is_duplicate")),
                    "gold_duplicate_of": expected.get("duplicate_of"),
                    "pred_duplicate_of": p.get("duplicate_of"),
                }
            )
        elif kind == "severity":
            sev_rows.append({"gold": expected.get("severity"), "pred": p.get("severity")})
        elif kind == "fix":
            fix_rows.append(
                {
                    "judge_score": pred.get("judge_score"),
                    "addresses_root_cause": pred.get("addresses_root_cause"),
                    "gold_can_fix": bool(expected.get("can_fix")),
                    "pred_can_fix": bool(p.get("can_fix")),
                }
            )

    return {
        "counts": {
            "dataset": len(dataset),
            "predicted": len([c for c in dataset if str(c["id"]) in preds]),
            "duplicate_family": len(dup_rows),
            "severity_family": len(sev_rows),
            "fix_family": len(fix_rows),
        },
        "duplicate": duplicate_metrics(dup_rows) if dup_rows else None,
        "severity": severity_metrics(sev_rows) if sev_rows else None,
        "fix": fix_metrics(fix_rows) if fix_rows else None,
        "ops": ops_metrics(ops_rows) if ops_rows else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute eval metrics from predictions.")
    parser.add_argument("--results-dir", default=str(_RESULTS_DIR))
    parser.add_argument(
        "--out", default=None, help="metrics.json path (default: results/metrics.json)"
    )
    args = parser.parse_args(argv)

    results = Path(args.results_dir)
    dataset = json.loads((results / "dataset.json").read_text(encoding="utf-8"))["cases"]
    predictions = json.loads((results / "predictions.json").read_text(encoding="utf-8"))[
        "predictions"
    ]
    metrics = compute_all(dataset, predictions)

    out = Path(args.out) if args.out else results / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps(metrics["counts"], indent=2))
    if metrics["duplicate"]:
        d = metrics["duplicate"]
        print(
            f"duplicate: P={d['precision']:.2f} R={d['recall']:.2f} F1={d['f1']:.2f} acc={d['accuracy']:.2f}"
        )
    if metrics["severity"]:
        s = metrics["severity"]
        print(
            f"severity: acc={s['accuracy']:.2f} adjacent={s['adjacent_accuracy']:.2f} (n={s['n']})"
        )
    if metrics["fix"]:
        f = metrics["fix"]
        print(f"fix: mean_score={f['mean_score']} reasonable%={f['pct_reasonable_or_better']:.2f}")


if __name__ == "__main__":
    main()
