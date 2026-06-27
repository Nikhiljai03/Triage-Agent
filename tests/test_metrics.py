"""Offline unit tests for the eval metric math (no network, no LLM, no Qdrant).

We hand-build small confusion matrices / score lists where the right answer is
obvious by inspection, then assert the functions reproduce them exactly. This is
the safety net that lets us trust the numbers in EVAL_REPORT.md.
"""

from __future__ import annotations

import math

from eval.metrics import (
    binary_prf,
    compute_all,
    confusion_matrix,
    duplicate_metrics,
    fix_metrics,
    ops_metrics,
    safe_div,
    severity_metrics,
)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, abs_tol=tol)


# --- primitives -------------------------------------------------------------
def test_safe_div_guards_zero_denominator() -> None:
    assert safe_div(5, 0) == 0.0
    assert safe_div(3, 4) == 0.75


def test_binary_prf_known_counts() -> None:
    # tp=3 fp=1 fn=2 tn=4 -> P=3/4, R=3/5, acc=7/10
    m = binary_prf(tp=3, fp=1, fn=2, tn=4)
    assert approx(m["precision"], 0.75)
    assert approx(m["recall"], 0.6)
    assert approx(m["f1"], 2 * 0.75 * 0.6 / (0.75 + 0.6))
    assert approx(m["accuracy"], 0.7)


# --- duplicate --------------------------------------------------------------
def test_duplicate_metrics_perfect() -> None:
    rows = [
        {
            "gold_is_duplicate": True,
            "pred_is_duplicate": True,
            "gold_duplicate_of": 1,
            "pred_duplicate_of": 1,
        },
        {
            "gold_is_duplicate": True,
            "pred_is_duplicate": True,
            "gold_duplicate_of": 2,
            "pred_duplicate_of": 2,
        },
        {
            "gold_is_duplicate": False,
            "pred_is_duplicate": False,
            "gold_duplicate_of": None,
            "pred_duplicate_of": None,
        },
    ]
    m = duplicate_metrics(rows)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0 and m["tn"] == 1
    assert approx(m["precision"], 1.0)
    assert approx(m["recall"], 1.0)
    assert approx(m["f1"], 1.0)
    assert approx(m["accuracy"], 1.0)
    assert approx(m["correct_identification"], 1.0)


def test_duplicate_metrics_mixed_and_wrong_target() -> None:
    rows = [
        # correct detection, but matched the wrong original issue id
        {
            "gold_is_duplicate": True,
            "pred_is_duplicate": True,
            "gold_duplicate_of": 10,
            "pred_duplicate_of": 99,
        },
        # missed a real duplicate (false negative)
        {
            "gold_is_duplicate": True,
            "pred_is_duplicate": False,
            "gold_duplicate_of": 11,
            "pred_duplicate_of": None,
        },
        # flagged a distinct issue as duplicate (false positive)
        {
            "gold_is_duplicate": False,
            "pred_is_duplicate": True,
            "gold_duplicate_of": None,
            "pred_duplicate_of": 5,
        },
        # correct reject (true negative)
        {
            "gold_is_duplicate": False,
            "pred_is_duplicate": False,
            "gold_duplicate_of": None,
            "pred_duplicate_of": None,
        },
    ]
    m = duplicate_metrics(rows)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert approx(m["precision"], 0.5)  # 1 tp / (1 tp + 1 fp)
    assert approx(m["recall"], 0.5)  # 1 tp / (1 tp + 1 fn)
    assert approx(m["accuracy"], 0.5)  # (1 + 1) / 4
    # the single TP matched the wrong original -> identification rate 0
    assert approx(m["correct_identification"], 0.0)


# --- severity ---------------------------------------------------------------
def test_confusion_matrix_buckets_unknown_pred() -> None:
    rows = [
        {"gold": "high", "pred": "high"},
        {"gold": "high", "pred": "low"},
        {"gold": "critical", "pred": None},  # inconclusive -> 'unknown' column
    ]
    cm = confusion_matrix(rows, ["critical", "high", "medium", "low"])
    assert cm["high"]["high"] == 1
    assert cm["high"]["low"] == 1
    assert cm["critical"]["unknown"] == 1
    assert cm["critical"]["critical"] == 0


def test_severity_metrics_exact_and_adjacent() -> None:
    rows = [
        {"gold": "critical", "pred": "critical"},  # exact
        {"gold": "critical", "pred": "high"},  # off-by-one (adjacent)
        {"gold": "high", "pred": "low"},  # off-by-two (not adjacent)
        {"gold": "medium", "pred": "medium"},  # exact
        {"gold": "low", "pred": None},  # unknown -> wrong, not adjacent
    ]
    m = severity_metrics(rows)
    assert m["n"] == 5
    assert m["exact_correct"] == 2
    assert approx(m["accuracy"], 2 / 5)
    # adjacent counts exact (2) + the critical->high off-by-one (1) = 3
    assert m["adjacent_correct"] == 3
    assert approx(m["adjacent_accuracy"], 3 / 5)
    # per-class: 'critical' predicted once correctly, predicted total once -> P=1, R=1/2
    assert approx(m["per_class"]["critical"]["precision"], 1.0)
    assert approx(m["per_class"]["critical"]["recall"], 0.5)
    assert m["per_class"]["critical"]["support"] == 2


# --- fix --------------------------------------------------------------------
def test_fix_metrics_scores_and_can_fix() -> None:
    rows = [
        {
            "judge_score": 5,
            "addresses_root_cause": True,
            "gold_can_fix": True,
            "pred_can_fix": True,
        },
        {
            "judge_score": 3,
            "addresses_root_cause": True,
            "gold_can_fix": True,
            "pred_can_fix": True,
        },
        {
            "judge_score": 1,
            "addresses_root_cause": False,
            "gold_can_fix": True,
            "pred_can_fix": True,
        },
        # should-escalate negative: no judge score, agent correctly declined to fix
        {
            "judge_score": None,
            "addresses_root_cause": None,
            "gold_can_fix": False,
            "pred_can_fix": False,
        },
    ]
    m = fix_metrics(rows)
    assert m["n"] == 4 and m["n_judged"] == 3
    assert approx(m["mean_score"], (5 + 3 + 1) / 3)
    assert m["median_score"] == 3
    assert approx(m["pct_reasonable_or_better"], 2 / 3)  # scores 5 and 3
    assert approx(m["pct_addresses_root_cause"], 2 / 3)
    assert approx(m["can_fix_accuracy"], 1.0)  # all four agree with gold
    assert m["score_distribution"][5] == 1 and m["score_distribution"][3] == 1


def test_fix_metrics_can_fix_mistakes() -> None:
    rows = [
        {
            "judge_score": 4,
            "addresses_root_cause": True,
            "gold_can_fix": True,
            "pred_can_fix": True,
        },
        # agent tried to fix something it should have escalated
        {
            "judge_score": None,
            "addresses_root_cause": None,
            "gold_can_fix": False,
            "pred_can_fix": True,
        },
    ]
    m = fix_metrics(rows)
    assert approx(m["can_fix_accuracy"], 0.5)


# --- ops --------------------------------------------------------------------
def test_ops_metrics_latency_tokens_cost() -> None:
    rows = [
        {
            "latency_s": 2.0,
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "llm_calls": 2,
        },
        {
            "latency_s": 4.0,
            "input_tokens": 3000,
            "output_tokens": 800,
            "total_tokens": 3800,
            "llm_calls": 3,
        },
    ]
    m = ops_metrics(rows, total_runtime_s=30.0)
    assert approx(m["mean_latency_s"], 3.0)
    assert approx(m["median_latency_s"], 3.0)
    assert m["total_llm_calls"] == 5
    assert m["total_tokens"] == 5000
    assert approx(m["mean_tokens_per_issue"], 2500.0)
    # cost: input 4000/1e6*0.30 + output 1000/1e6*2.50 = 0.0012 + 0.0025 = 0.0037
    assert approx(m["est_cost_total_usd_paid"], round(0.0037, 4))
    assert m["actual_cost_usd"] == 0.0
    assert m["total_runtime_s"] == 30.0


def test_ops_metrics_ignores_rows_without_usage() -> None:
    rows = [
        {"latency_s": 1.0, "total_tokens": None, "llm_calls": 0},
        {
            "latency_s": 3.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "llm_calls": 1,
        },
    ]
    m = ops_metrics(rows)
    # token mean is over the single row that has usage
    assert approx(m["mean_tokens_per_issue"], 150.0)
    assert approx(m["mean_latency_s"], 2.0)  # latency still averages both


# --- join -------------------------------------------------------------------
def test_compute_all_joins_and_skips_unpredicted() -> None:
    dataset = [
        {"id": "dup-1", "kind": "duplicate", "expected": {"is_duplicate": True, "duplicate_of": 7}},
        {"id": "dist-1", "kind": "distinct", "expected": {"is_duplicate": False}},
        {"id": "sev-1", "kind": "severity", "expected": {"severity": "high"}},
        {"id": "fix-1", "kind": "fix", "expected": {"can_fix": True}},
        {"id": "sev-2", "kind": "severity", "expected": {"severity": "low"}},  # unpredicted
    ]
    predictions = [
        {
            "id": "dup-1",
            "prediction": {"is_duplicate": True, "duplicate_of": 7},
            "latency_s": 1.0,
            "total_tokens": 100,
            "input_tokens": 80,
            "output_tokens": 20,
            "llm_calls": 1,
        },
        {
            "id": "dist-1",
            "prediction": {"is_duplicate": False},
            "latency_s": 1.0,
            "total_tokens": 90,
            "input_tokens": 70,
            "output_tokens": 20,
            "llm_calls": 1,
        },
        {
            "id": "sev-1",
            "prediction": {"severity": "high"},
            "latency_s": 2.0,
            "total_tokens": 200,
            "input_tokens": 150,
            "output_tokens": 50,
            "llm_calls": 1,
        },
        {
            "id": "fix-1",
            "prediction": {"can_fix": True},
            "judge_score": 4,
            "addresses_root_cause": True,
            "latency_s": 5.0,
            "total_tokens": 500,
            "input_tokens": 400,
            "output_tokens": 100,
            "llm_calls": 2,
        },
    ]
    m = compute_all(dataset, predictions)
    assert m["counts"]["dataset"] == 5
    assert m["counts"]["predicted"] == 4  # sev-2 was not predicted
    assert m["counts"]["duplicate_family"] == 2
    assert m["counts"]["severity_family"] == 1
    assert m["counts"]["fix_family"] == 1
    assert approx(m["duplicate"]["accuracy"], 1.0)
    assert approx(m["severity"]["accuracy"], 1.0)
    assert m["fix"]["n_judged"] == 1
    assert m["ops"]["n"] == 4
