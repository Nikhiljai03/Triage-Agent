"""Dependency bundle injected into the graph nodes.

Holding the LLM, run store, and tool callables here (rather than importing them
inside nodes) is what makes the whole agent testable offline: tests construct
``AgentDeps`` with fakes; production calls :meth:`AgentDeps.default`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent import tools
from agent.llm import BaseLLM


@dataclass
class AgentDeps:
    """Everything the nodes need to do their work."""

    llm: BaseLLM
    store: Any = None  # RunStore | None — live dashboard mirroring (None in tests)
    search_similar: Callable[..., list[dict[str, Any]]] = field(default=tools.search_similar_issues)
    run_sandbox: Callable[..., Any] = field(default=tools.run_in_sandbox)
    get_file: Callable[..., Any] = field(default=tools.get_file_contents)

    @classmethod
    def default(cls) -> AgentDeps:
        """Wire the real LLM + run store (used by the worker)."""
        from agent.llm import get_llm
        from shared.run_store import get_run_store

        return cls(llm=get_llm(), store=get_run_store())
