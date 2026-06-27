# Evaluation Report — Autonomous Issue-Triage Agent

_Generated 2026-06-27 · model `gemini-2.5-flash-lite` · 10/28 cases predicted._

## Methodology

All numbers below come from running the **real agent reasoning nodes** in
**dry-run** (no GitHub writes — the execution gate is never invoked, asserted at
startup) over a labeled set built entirely from real GitHub data. Model:
**`gemini-2.5-flash-lite`** (the agent is model-swappable by config; this run used the
model whose free-tier daily quota was available — `gemini-2.5-flash`'s free tier is
capped at 20 requests/day, so the same-family `gemini-2.5-flash-lite` was used).
Retrieval uses the production embedder + cross-encoder reranker
against a controlled Qdrant collection (`eval_issues`) holding
**36 issues**. Duplicate threshold (model confidence)
= **0.7**; retrieval similarity threshold =
**0.5**.

**Dataset — 28 cases** (6 duplicate, 6 distinct,
8 severity, 6 fixable, 2
should-escalate).

- **Duplicate / distinct** — source repo `pydantic/pydantic`. *Synthetic
  paraphrase method (disclosed):* for a `duplicate` case, an original issue is in
  the index and the input is an LLM-generated paraphrase of it; a correct system
  re-identifies the original. `distinct` negatives are real issues **held out** of
  the index that are not true duplicates of anything indexed (nearest indexed
  neighbour below 0.8 cosine); the **hardest** such negatives — those with the
  highest sub-cap similarity — are chosen, so precision is stress-tested against
  related-but-distinct issues rather than trivially satisfied by an empty
  candidate set.
- **Severity** — source repo `rust-lang/rust`, using its **real
  human-assigned priority labels** mapped 1:1 to our scale: `P-critical`→`critical`, `P-high`→`high`, `P-medium`→`medium`, `P-low`→`low`. No labels were
  invented. Severity is classified from issue text (sandbox repro is disabled for
  this cross-language set).
- **Fix quality** — source repo `pydantic/pydantic`. Cases are closed issues
  fixed by a **merged PR**; that PR's diff is the *reference fix*. The agent drafts
  a patch without seeing it, and an LLM judge scores draft-vs-reference for
  same-root-cause (1–5). The fix retriever **excludes the issue's own chunks**
  (leakage guard) so it cannot copy the reference. `should-escalate` negatives are
  `feature request` issues where the correct behaviour is to decline an auto-fix.

## Duplicate detection

| Metric | Value |
| --- | --- |
| Precision | 100.0% |
| Recall | 100.0% |
| F1 | 100.0% |
| Accuracy | 100.0% |
| Correct original identified (of TPs) | 100.0% (2/2) |
| Confusion (TP/FP/FN/TN) | 2/0/0/2 (n=4) |

Positive class = "this issue duplicates one already in the index". A predicted
duplicate counts only when model confidence ≥ the threshold.

## Severity classification

- **Exact accuracy:** 0.0% (0/2)
- **Adjacency (±1 rung) accuracy:** 100.0% (2/2) — severity is ordinal, so an off-by-one (e.g. `high` for a gold `critical`) is reported separately, never as exact-correct.

**Confusion matrix** (rows = gold priority label, cols = predicted):

| gold ＼ pred | critical | high | medium | low | unknown |
| --- | --- | --- | --- | --- | --- |
| **critical** | 0 | 2 | 0 | 0 | 0 |
| **high** | 0 | 0 | 0 | 0 | 0 |
| **medium** | 0 | 0 | 0 | 0 | 0 |
| **low** | 0 | 0 | 0 | 0 | 0 |

**Per-class** (one-vs-rest):

| class | precision | recall | support |
| --- | --- | --- | --- |
| critical | 0.0% | 0.0% | 2 |
| high | 0.0% | 0.0% | 0 |
| medium | 0.0% | 0.0% | 0 |
| low | 0.0% | 0.0% | 0 |

## Fix-draft quality

| Metric | Value |
| --- | --- |
| Cases judged (had a reference fix) | 2 |
| Mean judge score (1–5) | 4.0 |
| Median judge score | 4.0 |
| Rated reasonable-or-better (≥3) | 100.0% |
| Addresses root cause (≥4) | 50.0% |
| `can_fix` decision accuracy (both directions) | 50.0% (2/4) |
| Score distribution | 1★×0 · 2★×0 · 3★×1 · 4★×0 · 5★×1 |

Judge = same model (`gemini-2.5-flash-lite`) scoring the draft against the merged-PR diff.
Same-model judging is a known bias — see Limitations.

## Operational cost & latency

| Metric | Value |
| --- | --- |
| Mean latency / issue | 20.568s |
| Median latency / issue | 17.276s |
| Mean tokens / issue | 2045 (in 1810 / out 235) |
| Total LLM calls | 12 |
| Total tokens | 20,450 |
| Actual cost (Gemini free tier) | **$0.00** |
| Est. cost / issue at paid flash rates | $0.00113 |
| Est. total at paid flash rates | $0.0113 |
| Total wall-clock (incl. throttle) | — |

Paid-rate estimate uses representative gemini-2.5-flash prices ($0.30 / $2.50 per
1M input / output tokens); throttle delays inflate wall-clock but not token cost.

## Qualitative examples

**✅ Duplicate re-identified — `dup-13341`.** Paraphrase of pydantic#13341 matched back to it (confidence 0.95).

  - `retrieve_similar`: found 3 similar past issue(s)
  - `duplicate_check`: is_duplicate=True of #13341 conf=0.95 — The new issue directly describes the same problem as candidate issue #13341: the API documentation is missing the `**` indicator for keyword arguments, leading to confusion about method signatures. Both issues mention the impact on understanding and using the 
  - `propose`: [DRY-RUN] would label (label=duplicate) — matches #13341
  - `propose`: [DRY-RUN] would comment (body=This appears to be a duplicate of #13341. See that issue for the original report.) — link to #13341

**✅ Fix draft — `fix-13318`** (judge 5/5): _The agent's draft correctly identifies that `pathlib.WindowsPath` was missing from the type checking logic in `_generate_schema.py` and adds it, mirroring the reference fix's approach to include `WindowsPath` in the schema generation for path types._

  - `retrieve_fixes`: found 3 past fix(es) with diffs
  - `draft_fix`: can_fix=True conf=1.0 — The issue is that `WindowsPath` is not explicitly handled in `GenerateSchema.match_type`, causing it to be excluded from schema generation. Adding `WindowsPath` to the existing check for `Path` and `PosixPath` is a minimal and direct fix.
  - `propose`: [DRY-RUN] would draft_pr (title=Fix: Include WindowsPath in schema generation, body=The `GenerateSchema.match_type` method was excluding `pathlib.WindowsPath` from schema generation, while `pathlib.PosixPath` and `pathlib.Path` were included. This led to inconsistent schema generation across differe
  - `propose`: [DRY-RUN] would label (label=auto-fix-candidate) — agent drafted a minimal fix

**❌ Miss — `sev-rust-158378`:** predicted `high` vs gold `critical` (priority label `P-critical`).

  - `classify_severity`: severity=high conf=0.8 — The issue describes a consistent hang in CI for a specific build target (dist-i686-msvc) during the lint-docs process. While not a crash, a persistent hang in a CI job that prevents builds from completing is a significant problem affecting the development workflow and is lik

## Limitations

Disclosed plainly — these bound how far the numbers generalize:

0. **Partial run.** 10/28 cases predicted so far; the run paused on the Gemini free-tier daily cap (20 requests/day/model) and resumes from its checkpoint. Rates below will tighten as the remaining cases complete.
1. **Small sample.** A few dozen cases; treat every rate as indicative, not
   statistically tight. Per-class severity supports are single digits.
2. **Same-model judging.** Fix quality is scored by the same model that drafts
   (`gemini-2.5-flash-lite`); models tend to favour their own style. The judge prompt is
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
