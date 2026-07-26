"""Tests for the audit -> refine loop and its supervisor wiring.

The loop is a Mode 2 adapted port of AREX (arXiv:2607.21461v1): the outer
self-improvement loop that consumes the grounding audit's unsupported claims
and dispatches targeted follow-up research. These tests cover the loop in
isolation and its integration into the existing
:class:`~src.agents.supervisor.SupervisorAgent` -- the call site -- which is
what proves the wiring actually invokes the new code.
"""

import json

import pytest

from src.agents.audit_refine_loop import AuditRefineLoop, RefineOutcome
from src.agents.auditor import AuditResult
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
# AuditRefineLoop unit tests (new module)
# --------------------------------------------------------------------------- #


def test_derive_subqueries_keeps_salient_terms_drops_stopwords():
    loop = AuditRefineLoop()
    subqueries = loop.derive_followup_subqueries(
        ["The Mars colony population reached 9 million settlers by 2077."],
        research_query="space colonization",
    )
    assert len(subqueries) == 1
    query = subqueries[0]["query"]
    assert "mars" in query and "colony" in query and "population" in query
    # Magnitude / filler words are dropped from the query.
    assert "million" not in query
    assert "the" not in query
    assert subqueries[0]["priority"] == 1


def test_derive_subqueries_skips_claims_with_too_few_terms():
    loop = AuditRefineLoop(min_query_terms=3)
    # Only two non-stopword terms remain ("grew", "lot") -> nothing to target.
    subqueries = loop.derive_followup_subqueries(["It grew a lot."])
    assert subqueries == []


def test_derive_subqueries_dedupes_near_identical_claims():
    loop = AuditRefineLoop()
    claims = [
        "Mars colony population reached 9 million settlers",
        "Mars colony settlers population reached 9 million",  # same terms
    ]
    subqueries = loop.derive_followup_subqueries(claims)
    assert len(subqueries) == 1


def test_run_dispatches_retriever_with_derived_subqueries():
    loop = AuditRefineLoop()
    audit_result = AuditResult(
        unsupported_claims=["The Mars colony population reached 9 million settlers"]
    )
    dispatched = {}

    class StubRetriever:
        last_search_results = [
            {"url": "https://example.com/mars", "title": "Mars Colony"}
        ]

        def retrieve(self, query, subqueries_json):
            dispatched["query"] = query
            dispatched["subqueries"] = json.loads(subqueries_json)["subqueries"]
            return "findings"

    outcome = loop.run(
        audit_result=audit_result,
        research_query="space colonization",
        retriever=StubRetriever(),
        sources=[],
    )

    assert outcome.ran
    assert len(outcome.followup_subqueries) == 1
    assert "mars" in dispatched["subqueries"][0]["query"]
    assert outcome.new_sources[0]["url"] == "https://example.com/mars"


def test_run_noop_when_audit_passes():
    loop = AuditRefineLoop()
    audit_result = AuditResult()  # passed == True by default
    outcome = loop.run(
        audit_result=audit_result, research_query="q", retriever=None, sources=[]
    )
    assert not outcome.ran


def test_run_noop_when_audit_unverifiable():
    loop = AuditRefineLoop()
    # No sources were gathered -> audit could not surface specific gaps.
    audit_result = AuditResult(verifiable=False, unsupported_claims=["Mars colony"])
    outcome = loop.run(
        audit_result=audit_result,
        research_query="q",
        retriever=object(),
        sources=[],
    )
    assert not outcome.ran


def test_run_never_raises_on_retriever_failure():
    loop = AuditRefineLoop()
    audit_result = AuditResult(unsupported_claims=["Mars colony population settled"])

    class BrokenRetriever:
        last_search_results = []

        def retrieve(self, query, subqueries_json):
            raise RuntimeError("network down")

    outcome = loop.run(
        audit_result=audit_result,
        research_query="q",
        retriever=BrokenRetriever(),
        sources=[],
    )
    assert not outcome.ran  # best-effort: failure leaves report unrefined


def test_format_outcome_lists_subqueries_and_sources():
    loop = AuditRefineLoop()
    outcome = RefineOutcome(
        refined=True,
        followup_subqueries=[{"query": "mars colony population"}],
        new_sources=[{"url": "https://example.com/mars", "title": "Mars Colony"}],
    )
    rendered = loop.format_outcome(outcome)
    assert "Targeted Follow-up Research" in rendered
    assert "mars colony population" in rendered
    assert "https://example.com/mars" in rendered
    assert "arXiv:2607.21461" in rendered


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_refine_loop(patch_config_keys):
    supervisor = SupervisorAgent()
    assert isinstance(supervisor.refine_loop, AuditRefineLoop)


def test_supervisor_dispatches_followup_research_for_audit_gaps(patch_config_keys):
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = GROUNDING_SOURCES

    # Unsupported numeric claim (no source mentions Mars) -> audit flags a gap.
    report = "# Report\n\nThe Mars colony population reached 9 million settlers by 2077.\n"

    dispatched = []

    def fake_retrieve(query, subqueries_json):
        dispatched.append(subqueries_json)
        # Mimic the retriever exposing freshly gathered evidence.
        supervisor.web_search_retriever.last_search_results = [
            {
                "url": "https://example.com/mars-colony",
                "title": "Mars Colony Census",
                "text": "mars colony population 9 million settlers",
                "highlights": [],
            }
        ]
        return "Follow-up findings on the Mars colony population."

    supervisor.web_search_retriever.retrieve = fake_retrieve

    refined = supervisor._audit_report(report, research_query="space colonization")

    # The loop ran: a targeted subquery was derived from the gap and dispatched.
    assert dispatched, "supervisor should dispatch follow-up research for audit gaps"
    assert "Targeted Follow-up Research" in refined
    assert "AREX" in refined
    # New evidence was merged back into the gathered sources.
    assert any(
        s.get("url") == "https://example.com/mars-colony"
        for s in supervisor._gathered_sources
    )


def test_supervisor_skips_followup_when_report_is_grounded(patch_config_keys):
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = GROUNDING_SOURCES

    # Grounded report -> audit passes, nothing to follow up on.
    report = (
        "The quantum computing market is projected to reach 47 billion by 2030 "
        "as noted in [the market report](https://example.com/quantum-report)."
    )
    dispatched = []
    supervisor.web_search_retriever.retrieve = lambda q, s: dispatched.append(s) or "x"

    refined = supervisor._audit_report(report, research_query="quantum computing")

    assert not dispatched
    assert "Targeted Follow-up Research" not in refined
