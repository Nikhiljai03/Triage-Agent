"""Instrumented Gemini LLM for the eval harness: throttle, 429-retry, token meter.

Wraps the SAME model the agent uses (``GeminiLLM``) so the numbers reflect the
real system — production ``agent/llm.py`` stays untouched. Three jobs:

1. **Throttle** — sleep ``delay`` seconds before every call so a tight eval loop
   stays under Gemini free-tier per-minute limits.
2. **429 handling** — on ``RateLimitError`` sleep for the server-provided
   ``retryDelay`` (or a backoff) and retry, up to ``max_retries``. If the limit
   is *still* hit (e.g. the per-DAY cap is exhausted), raise
   :class:`RateLimitStop` — which derives from ``BaseException`` ON PURPOSE so the
   nodes' defensive ``except Exception`` cannot swallow it; it propagates to the
   harness, which checkpoints and stops cleanly instead of recording bogus
   "inconclusive" predictions.
3. **Metering** — tally calls, input/output/total tokens, and request seconds, so
   the report can state tokens/issue and an estimated paid cost.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass

from agent.llm import GeminiLLM, LLMError

logger = logging.getLogger("triage.eval.llm_meter")

_RETRY_DELAY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s")


class RateLimitStop(BaseException):
    """Raised when Gemini keeps rate-limiting after all retries (likely daily cap).

    Intentionally a ``BaseException`` (not ``Exception``) so the agent nodes'
    ``except Exception`` defensive handlers do NOT catch it — the eval harness
    catches it explicitly, checkpoints, and exits so the run can resume later.
    """


@dataclass
class Usage:
    """Cumulative usage counters (snapshotted/diffed per case by the harness)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_seconds: float = 0.0
    rate_limit_waits: int = 0


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    """Best-effort: pull a retry delay from the 429, else exponential backoff (cap 90s)."""
    # 1) Standard Retry-After header.
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    # 2) Gemini RESOURCE_EXHAUSTED body carries details[].retryDelay like "27s".
    body = getattr(exc, "body", None)
    text = json.dumps(body) if isinstance(body, (dict, list)) else (str(body) if body else str(exc))
    match = _RETRY_DELAY_RE.search(text or "")
    if match:
        return min(float(match.group(1)) + 1.0, 90.0)
    # 3) Fallback: exponential backoff.
    return min(5.0 * (2**attempt), 90.0)


class MeteredGeminiLLM(GeminiLLM):
    """``GeminiLLM`` + throttle + 429 retry + a token/latency meter."""

    def __init__(self, *args, delay: float = 0.0, max_retries: int = 6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.max_retries = max_retries
        self.usage = Usage()

    # -- metering helpers --------------------------------------------------
    def usage_snapshot(self) -> dict[str, float]:
        """Cumulative counters; the harness diffs two snapshots to get per-case usage."""
        return asdict(self.usage)

    @staticmethod
    def diff(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
        return {k: after[k] - before[k] for k in after}

    # -- the one method we override to instrument the request --------------
    def complete(self, system: str, user: str) -> str:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        if self.delay:
            time.sleep(self.delay)

        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except RateLimitError as exc:
                # Quota / rate limit (429): wait the server-provided delay, retry,
                # and on exhaustion STOP (likely the per-day cap) so we checkpoint.
                self.usage.rate_limit_waits += 1
                if attempt >= self.max_retries:
                    raise RateLimitStop(
                        f"Gemini rate limit not cleared after {self.max_retries} retries "
                        f"(likely the per-day free-tier cap). Last error: {exc}"
                    ) from None
                wait = _retry_after_seconds(exc, attempt)
                logger.warning(
                    "Gemini 429 (attempt %d/%d); sleeping %.1fs before retry.",
                    attempt + 1,
                    self.max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            except (APIStatusError, APIConnectionError) as exc:
                # Transient server hiccup (5xx / connection): short backoff + retry.
                status = getattr(exc, "status_code", None)
                if not (isinstance(exc, APIConnectionError) or status in (500, 502, 503, 504)):
                    raise LLMError(f"LLM request failed: {exc}") from exc
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"LLM request failed after {self.max_retries} retries: {exc}"
                    ) from exc
                wait = min(5.0 * (attempt + 1), 30.0)
                logger.warning(
                    "Gemini transient error %s (attempt %d/%d); sleeping %.1fs.",
                    status or "connection",
                    attempt + 1,
                    self.max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            except Exception as exc:  # noqa: BLE001 — match base: one clear error type.
                raise LLMError(f"LLM request failed: {exc}") from exc

            # Success — record usage and return.
            self.usage.request_seconds += time.monotonic() - started
            self.usage.calls += 1
            self._record_usage(resp)
            return resp.choices[0].message.content or ""

        raise LLMError("complete() exhausted retries without returning")  # unreachable

    def _record_usage(self, resp: object) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        self.usage.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.usage.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.usage.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
