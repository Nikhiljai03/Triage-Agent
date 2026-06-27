"""Assemble the triage state machine with LangGraph.

Flow::

    ingest_issue -> retrieve_similar -> duplicate_check
        duplicate_check --(duplicate, conf>=thresh)--> END
        duplicate_check --(otherwise)--> reproduce -> classify_severity -> retrieve_fixes
            retrieve_fixes --(severity in autofix set & confident)--> draft_fix
                draft_fix --(can_fix, conf>=thresh)--> END
                draft_fix --(otherwise)--> escalate -> END
            retrieve_fixes --(otherwise)--> escalate -> END

Every routing decision is a pure function of state fields the nodes set, so the
trace on the dashboard fully explains the path taken.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.deps import AgentDeps
from agent.state import TriageState, init_state
from shared.config import settings
from shared.schemas import IssueEvent


def _autofix_severities() -> set[str]:
    return {s.strip().lower() for s in settings.autofix_severities.split(",") if s.strip()}


# --- routers (pure functions of state) --------------------------------------
def route_after_duplicate(state: TriageState) -> str:
    conf = state.get("duplicate_confidence") or 0.0
    if state.get("is_duplicate") and conf >= settings.confidence_threshold:
        return "duplicate"
    return "investigate"


def route_after_fixes(state: TriageState) -> str:
    sev = state.get("severity")
    conf = state.get("severity_confidence") or 0.0
    if sev in _autofix_severities() and conf >= settings.confidence_threshold:
        return "draft"
    return "escalate"


def route_after_draft(state: TriageState) -> str:
    return "end" if state.get("drafted_fix") else "escalate"


def build_graph(deps: AgentDeps):
    """Compile the LangGraph state machine, binding each node to ``deps``."""
    graph = StateGraph(TriageState)

    for name in (
        "ingest_issue",
        "retrieve_similar",
        "duplicate_check",
        "reproduce",
        "classify_severity",
        "retrieve_fixes",
        "draft_fix",
        "escalate",
    ):
        graph.add_node(name, partial(getattr(nodes, name), deps=deps))

    graph.add_edge(START, "ingest_issue")
    graph.add_edge("ingest_issue", "retrieve_similar")
    graph.add_edge("retrieve_similar", "duplicate_check")
    graph.add_conditional_edges(
        "duplicate_check", route_after_duplicate, {"duplicate": END, "investigate": "reproduce"}
    )
    graph.add_edge("reproduce", "classify_severity")
    graph.add_edge("classify_severity", "retrieve_fixes")
    graph.add_conditional_edges(
        "retrieve_fixes", route_after_fixes, {"draft": "draft_fix", "escalate": "escalate"}
    )
    graph.add_conditional_edges(
        "draft_fix", route_after_draft, {"end": END, "escalate": "escalate"}
    )
    graph.add_edge("escalate", END)
    return graph.compile()


def run_triage(event: IssueEvent, run_id: str, deps: AgentDeps | None = None) -> TriageState:
    """Run the full triage graph for one issue and return the final state."""
    deps = deps or AgentDeps.default()
    graph = build_graph(deps)
    return graph.invoke(init_state(event, run_id))
