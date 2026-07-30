"""Tests for the relay-flow coordination analyzer and its supervisor wiring.

The analyzer is adapted from *When Do Multi-Agent Systems Help? An Information
Bottleneck Perspective* (arXiv:2607.16133v1). These tests cover the analyzer in
isolation and its integration into the existing
:class:`~src.agents.supervisor.SupervisorAgent` -- the call site -- which is
what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.relay_flow import RelayFlowAnalyzer
from src.agents.research_trace import ResearchTrace
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# Each source carries 100 chars of text so gathered-text math is exact.
_TEXT = "x" * 100

# Flat sources (4 gathered; report will cite 3 of them).
FLAT_SOURCES = [
    {"url": "https://example.com/a", "title": "", "text": _TEXT},
    {"url": "https://example.com/b", "title": "", "text": _TEXT},
    {"url": "https://example.com/c", "title": "", "text": _TEXT},
    {"url": "https://example.com/d", "title": "", "text": _TEXT},
]

# Nested subquery buckets, matching WebSearchRetriever.last_search_results shape.
NESTED_SOURCES = [
    {
        "subquery": "market size",
        "priority": 1,
        "results": [
            {"url": "https://example.com/a", "title": "A", "text": "alpha " * 40},
            {"url": "https://example.com/b", "title": "B", "text": "beta " * 40},
        ],
        "similar_results": [
            {"url": "https://example.com/c", "title": "C", "text": "gamma " * 40},
        ],
    }
]


def _full_trace() -> ResearchTrace:
    """A trace with the full subgoal -> tool -> tool -> evidence -> claim chain."""
    trace = ResearchTrace()
    trace.record_subgoal("What is X?")
    trace.record_tool("planning_agent", {"research_query": "What is X?"})
    trace.record_tool("web_search_retriever", {"research_query": "What is X?"})
    trace.record_evidence(FLAT_SOURCES)
    return trace


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# RelayFlowAnalyzer unit tests (new module)
# --------------------------------------------------------------------------- #


def test_favorable_when_reduction_outweighs_loss():
    analyzer = RelayFlowAnalyzer()
    # 4 sources (400 chars); report cites 3 -> retention 0.75, loss 0.25.
    report = (
        "Findings backed by [a](https://example.com/a), "
        "[b](https://example.com/b) and [c](https://example.com/c)."
    )
    result = analyzer.analyze(_full_trace(), FLAT_SOURCES, report)

    assert result.verifiable is True
    assert result.n_sources_gathered == 4
    assert result.n_sources_retained == 3
    assert result.retention == pytest.approx(0.75)
    assert result.information_loss == pytest.approx(0.25)
    # gathered 400 chars vs ~100-char report => ~4x compression, benefit ~0.75.
    assert result.reduction_benefit > result.information_loss
    assert result.favorable is True
    assert result.effective_beta > 0.0


def test_lossy_when_nothing_survives_the_relays():
    analyzer = RelayFlowAnalyzer()
    # Report cites none of the gathered sources.
    report = "Generic prose with no inline citations at all."
    result = analyzer.analyze(_full_trace(), FLAT_SOURCES, report)

    assert result.n_sources_retained == 0
    assert result.retention == 0.0
    assert result.information_loss == 1.0
    assert result.favorable is False
    assert "dropped all gathered evidence" in result.verdict


def test_url_normalization_matches_www_and_scheme():
    analyzer = RelayFlowAnalyzer()
    sources = [{"url": "http://www.example.com/page/", "text": "x" * 100}]
    report = "See [page](https://example.com/page)."
    trace = ResearchTrace()
    trace.record_subgoal("q")
    trace.record_tool("web_search_retriever", {"research_query": "q"})
    trace.record_evidence(sources)

    result = analyzer.analyze(trace, sources, report)

    assert result.n_sources_retained == 1
    assert result.retention == 1.0


def test_no_evidence_is_unverifiable_not_penalized():
    analyzer = RelayFlowAnalyzer()
    result = analyzer.analyze(_full_trace(), [], "See [a](https://x.io/y).")

    assert result.verifiable is False
    assert result.favorable is False
    assert "cannot be judged" in result.verdict


def test_counts_subquery_buckets_for_planning_relay():
    analyzer = RelayFlowAnalyzer()
    trace = ResearchTrace()
    trace.record_subgoal("q")
    trace.record_tool("planning_agent", {"research_query": "q"})
    trace.record_tool("web_search_retriever", {"research_query": "q"})
    trace.record_evidence(NESTED_SOURCES)
    report = "[a](https://example.com/a) [b](https://example.com/b) [c](https://example.com/c)"

    result = analyzer.analyze(trace, NESTED_SOURCES, report)

    # One bucket -> one subquery searched; three sources flattened.
    assert result.n_subqueries == 1
    assert result.n_sources_gathered == 3
    assert result.n_sources_retained == 3
    # R2 planning relay and R3/R4 measurable relays are all present.
    names = [hop.name for hop in result.hops]
    assert names == ["R1", "R2", "R3", "R4"]
    r4 = next(h for h in result.hops if h.name == "R4")
    assert r4.loss == 0.0


def test_format_report_renders_section_and_metrics():
    analyzer = RelayFlowAnalyzer()
    report = (
        "Findings backed by [a](https://example.com/a) and "
        "[b](https://example.com/b)."
    )
    result = analyzer.analyze(_full_trace(), FLAT_SOURCES, report, model_name="m2.1")
    rendered = analyzer.format_report(result)

    assert "## Multi-Agent Coordination Analysis" in rendered
    assert "arXiv:2607.16133" in rendered
    assert "effective beta" in rendered
    # The R4 relay reports the gathered-vs-retained evidence.
    assert "2/4 gathered source(s) cited" in rendered
    assert "m2.1" in rendered  # capability proxy surfaced


def test_format_report_unverifiable_path():
    analyzer = RelayFlowAnalyzer()
    result = analyzer.analyze(_full_trace(), [], "no sources")
    rendered = analyzer.format_report(result)

    assert "could not be judged" in rendered


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_relay_flow(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.relay_flow, RelayFlowAnalyzer)


def test_supervisor_analyze_coordination_appends_section(patch_config_keys):
    """Drive the supervisor's coordination-analysis path and assert the section
    is appended. We reproduce what SupervisorAgent.research() records on a run
    -- seed the subgoal, the tool calls, the evidence -- then invoke the wired
    helper, with no network call, so the wiring is what is under test.
    """
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = NESTED_SOURCES

    supervisor.trace.reset()
    supervisor.trace.record_subgoal("What is X?")
    supervisor.trace.record_tool("planning_agent", {"research_query": "What is X?"})
    supervisor.trace.record_tool(
        "web_search_retriever", {"research_query": "What is X?"}
    )
    supervisor.trace.record_evidence(NESTED_SOURCES)

    report = (
        "# Report\n\n"
        "Findings from [A](https://example.com/a) and [B](https://example.com/b).\n"
    )
    supervisor.trace.record_report(report)

    analyzed = supervisor._analyze_coordination(report)

    # Original report is preserved and the coordination section is appended.
    assert analyzed.startswith("# Report")
    assert "## Multi-Agent Coordination Analysis" in analyzed
    assert "arXiv:2607.16133" in analyzed
    # 2 of 3 gathered sources surfaced in the report.
    assert "2/3 gathered source(s) cited" in analyzed


def test_supervisor_analyze_coordination_skips_cleanly_with_no_sources(
    patch_config_keys,
):
    """No sources captured -> unverifiable, but the report is returned intact."""
    supervisor = SupervisorAgent()
    supervisor.trace.reset()
    supervisor.trace.record_subgoal("What is X?")
    supervisor.trace.record_report("# Report\n\nFindings.")

    report = "# Report\n\nFindings with [a link](https://x.io/y)."
    analyzed = supervisor._analyze_coordination(report)

    assert analyzed.startswith("# Report")
    assert "could not be judged" in analyzed
