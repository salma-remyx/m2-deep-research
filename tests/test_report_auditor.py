"""Tests for the report grounding auditor and its supervisor wiring.

The auditor is adapted from BrainPilot's Auditor agent (arXiv:2607.15079v1).
These tests cover both the auditor in isolation and its integration into the
existing :class:`~src.agents.supervisor.SupervisorAgent` -- the call site --
which is what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.auditor import AuditResult, ReportAuditor
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# Flat source fixtures (mirror ExaTool.format_results output).
GROUNDING_SOURCES = [
    {
        "title": "Quantum Computing Market Report",
        "url": "https://example.com/quantum-report",
        "text": "The quantum computing market is projected to reach 47 billion "
        "dollars by 2030 according to industry analysts.",
        "highlights": ["market projected to reach 47 billion"],
    },
    {
        "title": "Qubit Fidelity Research",
        "url": "http://www.example.com/qubit-fidelity/",
        "text": "Error rates in superconducting qubits have dropped below one "
        "percent over the last year.",
        "highlights": ["superconducting qubits error rates"],
    },
]


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# Auditor unit tests (new module)
# --------------------------------------------------------------------------- #


def test_flags_fabricated_citation():
    auditor = ReportAuditor()
    report = (
        "Growth is strong [Grand View](https://example.com/quantum-report) "
        "and also [made up](https://example.com/fabricated)."
    )
    result = auditor.audit(report, GROUNDING_SOURCES)

    assert result.verifiable is True
    assert result.total_citations == 2
    assert result.grounded_citations == 1
    assert "https://example.com/fabricated" in result.unsupported_citations
    assert result.score == pytest.approx(0.5)


def test_url_normalization_matches_www_and_scheme():
    # source uses http + www + trailing slash; citation uses clean https url
    auditor = ReportAuditor()
    report = "See [fidelity](https://example.com/qubit-fidelity)."
    result = auditor.audit(report, GROUNDING_SOURCES)

    assert result.unsupported_citations == []
    assert result.score == 1.0


def test_grounded_report_passes_clean():
    auditor = ReportAuditor()
    report = (
        "The quantum computing market is projected to reach 47 billion by 2030 "
        "as noted in [the market report](https://example.com/quantum-report)."
    )
    result = auditor.audit(report, GROUNDING_SOURCES)

    assert result.passed is True
    assert result.unsupported_citations == []
    assert result.unsupported_claims == []


def test_flags_unsupported_numeric_claim():
    auditor = ReportAuditor()
    # Numbers/words that appear in no retrieved source.
    report = "The Mars colony population reached 9 million settlers by 2077."
    result = auditor.audit(report, GROUNDING_SOURCES)

    assert any("Mars" in claim for claim in result.unsupported_claims)


def test_grounded_claim_not_flagged():
    auditor = ReportAuditor()
    report = "The quantum computing market is projected to reach 47 billion by 2030."
    result = auditor.audit(report, GROUNDING_SOURCES)

    assert result.unsupported_claims == []


def test_empty_sources_is_unverifiable_not_penalized():
    auditor = ReportAuditor()
    result = auditor.audit("See [a link](https://x.io/y).", [])

    assert result.verifiable is False
    assert result.score == 1.0
    assert result.unsupported_citations == []


def test_handles_nested_retriever_buckets():
    """Production shape: WebSearchRetriever returns subquery buckets, not flat sources."""
    auditor = ReportAuditor()
    nested = [
        {
            "subquery": "market size",
            "priority": 1,
            "results": [GROUNDING_SOURCES[0]],
            "similar_results": [],
        }
    ]
    report = "Backed by [the report](https://example.com/quantum-report)."
    result = auditor.audit(report, nested)

    assert result.sources_checked == 1
    assert result.unsupported_citations == []


def test_format_report_mentions_fabricated_url():
    auditor = ReportAuditor()
    report = "[real](https://example.com/quantum-report) [fake](https://x.io/nope)"
    result = auditor.audit(report, GROUNDING_SOURCES)
    rendered = auditor.format_report(result)

    assert "Source Grounding Audit" in rendered
    assert "https://x.io/nope" in rendered
    assert "arXiv:2607.15079" in rendered


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_auditor(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.auditor, ReportAuditor)
    assert supervisor._gathered_sources == []


def test_supervisor_audit_appends_findings_to_report(patch_config_keys):
    supervisor = SupervisorAgent()
    # Simulate sources captured from a web_search_retriever tool call.
    supervisor._gathered_sources = GROUNDING_SOURCES

    report = (
        "# Report\n\n"
        "Quantum growth is real [report](https://example.com/quantum-report) "
        "but [fabricated](https://example.com/made-up) is not backed.\n"
    )
    audited = supervisor._audit_report(report)

    # Original report is preserved and the audit section is appended.
    assert audited.startswith("# Report")
    assert "Source Grounding Audit" in audited
    assert "https://example.com/made-up" in audited


def test_supervisor_audit_skips_cleanly_with_no_sources(patch_config_keys):
    supervisor = SupervisorAgent()
    # No sources captured -> unverifiable, but the report is still returned intact.
    report = "# Report\n\nSome findings with [a link](https://x.io/y)."
    audited = supervisor._audit_report(report)

    assert audited.startswith("# Report")
    assert "could not be verified" in audited


def test_execute_tool_captures_retrieved_sources(patch_config_keys):
    supervisor = SupervisorAgent()

    captured = [
        {"url": "https://example.com/stub", "title": "Stub", "text": "stub", "highlights": []}
    ]
    # Stub the retriever so no network call is made.
    supervisor.web_search_retriever.retrieve = lambda query, subq: "findings"
    supervisor.web_search_retriever.last_search_results = captured

    out = supervisor.execute_tool(
        "web_search_retriever",
        {"research_query": "q", "subqueries_json": '{"subqueries": []}'},
    )

    assert out == "findings"
    assert supervisor._gathered_sources == captured
