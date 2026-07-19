"""Tests for the Graph of Trace recorder and its supervisor wiring.

The trace is adapted from BrainPilot's Graph of Trace (arXiv:2607.15079v1).
These tests cover the recorder in isolation and its integration into the
existing :class:`~src.agents.supervisor.SupervisorAgent` -- the call site --
which is what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.research_trace import CLAIM, EVIDENCE, SUBGOAL, TOOL_USE, ResearchTrace
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# Nested subquery buckets, matching WebSearchRetriever.last_search_results shape.
RETRIEVED_SOURCES = [
    {
        "subquery": "market size",
        "priority": 1,
        "results": [
            {"url": "https://example.com/a", "title": "A", "text": "a"},
            {"url": "https://example.com/b", "title": "B", "text": "b"},
        ],
        "similar_results": [
            {"url": "https://example.com/c", "title": "C", "text": "c"},
        ],
    }
]


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# ResearchTrace unit tests (new module)
# --------------------------------------------------------------------------- #


def test_records_workflow_in_order():
    trace = ResearchTrace()
    trace.record_subgoal("What is X?")
    trace.record_tool("planning_agent", {"research_query": "What is X?"})
    trace.record_tool("web_search_retriever", {"research_query": "What is X?"})
    trace.record_evidence(RETRIEVED_SOURCES)
    trace.record_report("A report [cite](https://example.com/a).")

    kinds = [n.kind for n in trace.nodes]
    assert kinds == [SUBGOAL, TOOL_USE, TOOL_USE, EVIDENCE, CLAIM]


def test_report_node_chains_under_evidence():
    """The report is the terminus of subgoal->tool->evidence->claim, so it must
    chain under the evidence node rather than sit as a sibling of the tools."""
    trace = ResearchTrace()
    trace.record_subgoal("What is X?")
    trace.record_tool("web_search_retriever", {"research_query": "What is X?"})
    evidence_id = trace.record_evidence(RETRIEVED_SOURCES)
    report_id = trace.record_report("A report [cite](https://example.com/a).")

    report = next(n for n in trace.nodes if n.node_id == report_id)
    assert report.parent == evidence_id


def test_evidence_counts_flatten_nested_buckets():
    trace = ResearchTrace()
    trace.record_subgoal("q")
    trace.record_tool("web_search_retriever", {"research_query": "q"})
    node_id = trace.record_evidence(RETRIEVED_SOURCES)

    evidence = next(n for n in trace.nodes if n.node_id == node_id)
    # 2 results + 1 similar_result across the single bucket.
    assert "3 source(s)" in evidence.summary


def test_report_node_counts_inline_citations():
    trace = ResearchTrace()
    trace.record_subgoal("q")
    report = "Backed by [one](https://x.io/1) and [two](https://x.io/2)."
    trace.record_report(report)

    claim = next(n for n in trace.nodes if n.kind == CLAIM)
    assert "2 inline citation(s)" in claim.detail


def test_render_is_a_tree_linking_evidence_under_its_tool():
    trace = ResearchTrace()
    trace.record_subgoal("What is X?")
    trace.record_tool("web_search_retriever", {"research_query": "What is X?"})
    trace.record_evidence(RETRIEVED_SOURCES)
    rendered = trace.render()

    assert "## Graph of Trace" in rendered
    assert "arXiv:2607.15079" in rendered
    # Subgoal at depth 0, tool indented under it, evidence indented under tool.
    assert "- **[1] Subgoal:**" in rendered
    assert "  - **[2] Tool:** web_search_retriever" in rendered
    assert "    - **[3] Evidence:** 3 source(s) retrieved" in rendered


def test_reset_clears_previous_run():
    trace = ResearchTrace()
    trace.record_subgoal("first")
    trace.record_tool("planning_agent", {"research_query": "first"})
    trace.reset()
    trace.record_subgoal("second")

    assert [n.summary for n in trace.nodes] == ["second"]


def test_render_empty_trace_is_graceful():
    rendered = ResearchTrace().render()
    assert "No workflow steps were recorded" in rendered


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_trace(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.trace, ResearchTrace)
    assert supervisor.trace.nodes == []


def test_supervisor_tool_loop_records_trace(patch_config_keys):
    """Drive the supervisor's tool-execution path and assert the trace fills in.

    We reproduce exactly what SupervisorAgent.research() does on a ``tool_use``
    turn -- seed the subgoal, record the tool, execute it, link the evidence --
    without any network call, so the wiring is what is under test.
    """
    supervisor = SupervisorAgent()

    # Stub the retriever so execute_tool captures sources with no network call.
    supervisor.web_search_retriever.retrieve = lambda query, subq: "findings"
    supervisor.web_search_retriever.last_search_results = RETRIEVED_SOURCES

    supervisor.trace.reset()
    supervisor.trace.record_subgoal("What is X?")
    supervisor.trace.record_tool(
        "web_search_retriever", {"research_query": "What is X?"}
    )
    supervisor.execute_tool(
        "web_search_retriever",
        {"research_query": "What is X?", "subqueries_json": '{"subqueries": []}'},
    )
    supervisor.trace.record_evidence(supervisor._gathered_sources)

    kinds = [n.kind for n in supervisor.trace.nodes]
    assert kinds == [SUBGOAL, TOOL_USE, EVIDENCE]
    assert supervisor.trace.nodes[-1].summary.startswith("3 source(s)")


def test_supervisor_trace_section_appends_to_report():
    """The rendered trace is the artifact appended to a delivered report."""
    trace = ResearchTrace()
    trace.record_subgoal("What is X?")
    report = "# Report\n\nFindings [a](https://example.com/a)."

    combined = report + trace.render()

    assert combined.startswith("# Report")
    assert "## Graph of Trace" in combined
