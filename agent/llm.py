"""LLM interface + a Gemini-backed implementation (swappable by config).

The agent reasons through Gemini's **OpenAI-compatible** endpoint, so we can use
the plain ``openai`` SDK pointed at Gemini's base URL. Swapping to OpenAI/Ollama
later is a config change (base_url/model/key), not a code change — keep all
provider specifics inside :class:`GeminiLLM` and behind :func:`get_llm`.

``complete_json`` is defensive: it instructs the model to emit JSON only, strips
`````json`` fences, and retries once with a nudge before giving up — callers
(the nodes) catch :class:`LLMError`/parse failures and escalate rather than crash.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from shared.config import settings

logger = logging.getLogger("triage.agent.llm")

# Deterministic-ish triage: low temperature, bounded output.
_TEMPERATURE = 0.1
_MAX_TOKENS = 2048


class LLMError(RuntimeError):
    """Raised when the LLM call or response handling fails irrecoverably."""


def _strip_to_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences/prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence (``` or ```json) and the trailing fence.
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    # Fall back to the outermost {...} span if there's leading/trailing prose.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


class BaseLLM(ABC):
    """Interface every LLM backend implements. ``complete_json`` is shared."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text completion for a system+user prompt."""

    def complete_json(self, system: str, user: str, schema_hint: str = "") -> dict[str, Any]:
        """Return parsed JSON from the model, retrying once on malformed output."""
        instruction = (
            f"{system}\n\nReturn ONLY a single valid JSON object"
            + (f" matching this shape: {schema_hint}" if schema_hint else "")
            + ". No markdown, no code fences, no prose."
        )
        raw = self.complete(instruction, user)
        try:
            return _strip_to_json(raw)
        except (ValueError, json.JSONDecodeError):
            logger.warning("LLM returned non-JSON; retrying once with a nudge.")
            retry = self.complete(
                instruction + "\n\nYour previous reply was not valid JSON. Output JSON only.",
                user,
            )
            try:
                return _strip_to_json(retry)
            except (ValueError, json.JSONDecodeError) as exc:
                raise LLMError(f"model did not return valid JSON: {exc}") from exc


class GeminiLLM(BaseLLM):
    """Gemini via its OpenAI-compatible endpoint (uses the ``openai`` SDK)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = _TEMPERATURE,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        from openai import OpenAI  # lazy: keep import cost out of test collection

        self.model = model or settings.llm_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key or settings.gemini_api_key,
            base_url=base_url or settings.gemini_base_url,
        )

    def complete(self, system: str, user: str) -> str:
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
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — surface a single clear error type.
            raise LLMError(f"LLM request failed: {exc}") from exc


def get_llm() -> BaseLLM:
    """Factory for the active LLM backend (Gemini today; swap via config)."""
    return GeminiLLM()
