"""Tests for the typed evidence graph and its supervisor wiring.

The evidence graph is adapted from EviGraph (arXiv:2608.04738v1). These tests
cover the graph in isolation and its integration into the existing
:class:`~src.agents.supervisor.SupervisorAgent` -- the call site -- which is
what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.evidence_graph import (
    CLAIM,
    EVIDENCE,
    MISSING_DEPENDENCY,
    PROBLEM,
    RESULT_CLAIM_INCONSISTENCY,
    SEMANTIC_MISALIGNMENT,
    EvidenceGraph,
)
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
# EvidenceGraph unit tests (new module)
# --------------------------------------------------------------------------- #


def test_builds_typed_graph_from_run():
    report = (
        "The quantum computing market is projected to reach 47 billion by 2030 "
        "[report](https://example.com/quantum-report)."
    )
    graph = EvidenceGraph.from_run(report, GROUNDING_SOURCES, query="quantum market")

    counts = graph.node_counts()
    assert counts == {PROBLEM: 1, EVIDENCE: 2, CLAIM: 1}
    # The claim is linked to the evidence it cites.
    claim = next(n for n in graph.nodes if n.kind == CLAIM)
    assert claim.support is not None
    assert claim.parent == claim.support


def test_well_grounded_claim_has_no_defect():
    report = (
        "The quantum computing market is projected to reach 47 billion by 2030 "
        "[report](https://example.com/quantum-report)."
    )
    graph = EvidenceGraph.from_run(report, GROUNDING_SOURCES)
    result = graph.validate()

    assert result.total_claims == 1
    assert result.supported_claims == 1
    assert result.weak_nodes == []
    assert result.support_rate == 1.0


def test_flags_cited_url_not_in_evidence():
    """A claim that cites a URL absent from gathered evidence has no real support."""
    report = "Growth is strong [made up](https://example.com/fabricated)."
    graph = EvidenceGraph.from_run(report, GROUNDING_SOURCES)
    result = graph.validate()

    claim = result.earliest_weak_node
    assert claim is not None
    assert MISSING_DEPENDENCY in claim.defect_categories
    assert RESULT_CLAIM_INCONSISTENCY in claim.defect_categories


def test_semantic_misalignment_against_cited_source_only():
    """Citing source A while talking about source B's content is misalignment.

    This is the graph's distinct signal: the auditor checks a claim against the
    whole corpus, whereas the graph checks it against the *specific* evidence it
    cites, so 'cited the wrong source' surfaces here.
    """
    sources = [
        {"url": "https://example.com/apple", "title": "Apple", "text": "Apple revenue grew 10 percent."},
        {"url": "https://example.com/tesla", "title": "Tesla", "text": "Tesla delivered 500 cars in 2024."},
    ]
    # Cites Apple but the claim's substance is Tesla's cars/2024 delivery.
    report = "Apple shipped 500 cars in 2024 [apple source](https://example.com/apple)."
    graph = EvidenceGraph.from_run(report, sources)
    result = graph.validate()

    claim = result.earliest_weak_node
    assert claim is not None
    assert SEMANTIC_MISALIGNMENT in claim.defect_categories
    # It IS linked to evidence (Apple exists), so it is not a missing dependency.
    assert MISSING_DEPENDENCY not in claim.defect_categories


def test_no_evidence_flags_every_claim_as_missing_dependency():
    """The strongest structural signal: claims exist but no evidence was gathered.

    Unlike the flat auditor (which returns score 1.0 and does not penalize when
    nothing was retrieved), the graph reports a 0% support rate.
    """
    report = "The Mars colony population reached 9 million settlers by 2077."
    graph = EvidenceGraph.from_run(report, [])
    result = graph.validate()

    assert result.evidence_count == 0
    assert result.total_claims == 1
    assert result.supported_claims == 0
    assert result.support_rate == 0.0
    assert MISSING_DEPENDENCY in result.earliest_weak_node.defect_categories


def test_earliest_weak_node_is_first_in_document_order():
    report = (
        "First [fake](https://example.com/fabricated) claim. "
        "The quantum computing market is projected to reach 47 billion by 2030 "
        "[report](https://example.com/quantum-report)."
    )
    graph = EvidenceGraph.from_run(report, GROUNDING_SOURCES)
    result = graph.validate()

    assert len(result.weak_nodes) == 1
    assert result.earliest_weak_node is result.weak_nodes[0]
    assert "First" in result.earliest_weak_node.summary


def test_support_rate_is_fraction_of_supported_claims():
    report = (
        "Real quantum growth to 47 billion [report](https://example.com/quantum-report). "
        "Fake claim [made up](https://example.com/fabricated)."
    )
    graph = EvidenceGraph.from_run(report, GROUNDING_SOURCES)
    result = graph.validate()

    assert result.total_claims == 2
    assert result.supported_claims == 1
    assert result.support_rate == pytest.approx(0.5)


def test_checkpoint_protects_validated_state_from_failed_repair():
    """A checkpoint survives a downstream mutation that a restore rolls back."""
    report = "The quantum computing market is projected to reach 47 billion by 2030."
    graph = EvidenceGraph.from_run(report, GROUNDING_SOURCES)
    graph.validate()
    validated_count = len(graph.nodes)
    # Checkpoint captures the current node count.
    assert graph.checkpoint() == validated_count

    # Simulate an unsuccessful repair: a new bogus claim is introduced.
    graph.add_claim("bogus repair claim [fake](https://example.com/zzz)")
    assert len(graph.nodes) == validated_count + 1

    assert graph.restore() is True
    assert len(graph.nodes) == validated_count
    assert graph.restore() is True  # snapshot still available for repeated restore


def test_checkpoint_without_snapshot_does_not_restore():
    graph = EvidenceGraph()
    assert graph.restore() is False


def test_handles_nested_retriever_buckets():
    """Production shape: WebSearchRetriever returns subquery buckets, not flat sources."""
    nested = [
        {
            "subquery": "market size",
            "priority": 1,
            "results": [GROUNDING_SOURCES[0]],
            "similar_results": [],
        }
    ]
    report = "Quantum market 47 billion [report](https://example.com/quantum-report)."
    graph = EvidenceGraph.from_run(report, nested)

    assert graph.node_counts()[EVIDENCE] == 1
    result = graph.validate()
    assert result.supported_claims == 1


def test_empty_report_is_graceful():
    graph = EvidenceGraph.from_run("# Title only", GROUNDING_SOURCES)
    result = graph.validate()

    assert result.total_claims == 0
    assert result.support_rate == 1.0
    assert result.earliest_weak_node is None


def test_format_report_mentions_rate_and_earliest_weak_node():
    report = "Growth is strong [made up](https://example.com/fabricated)."
    graph = EvidenceGraph.from_run(report, GROUNDING_SOURCES)
    rendered = graph.format_report(graph.validate())

    assert "## Evidence Graph Validation" in rendered
    assert "arXiv:2608.04738" in rendered
    assert "Claim support rate:" in rendered
    assert "Earliest weak node" in rendered
    assert "Result-claim inconsistency" in rendered


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_audit_report_appends_evidence_graph_section(patch_config_keys):
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = GROUNDING_SOURCES

    report = (
        "# Report\n\n"
        "Real quantum growth to 47 billion [report](https://example.com/quantum-report) "
        "but [fabricated](https://example.com/made-up) is not backed.\n"
    )
    audited = supervisor._audit_report(report)

    # Original report is preserved and both audit sections are appended.
    assert audited.startswith("# Report")
    assert "Source Grounding Audit" in audited
    assert "## Evidence Graph Validation" in audited
    assert "Claim support rate:" in audited


def test_supervisor_evidence_graph_uses_research_query(patch_config_keys):
    """The query recovered from history becomes the graph's Problem node."""
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = GROUNDING_SOURCES
    supervisor.messages = [{"role": "user", "content": "quantum computing market"}]

    report = "The market reached 47 billion [report](https://example.com/quantum-report)."
    audited = supervisor._audit_report(report)

    # Problem node count surfaces in the rendered typed-graph line.
    assert "1 problem" in audited


def test_supervisor_evidence_graph_runs_even_with_no_sources(patch_config_keys):
    """No sources -> the graph still runs and flags the structural gap."""
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = []

    report = "# Report\n\nSome ungrounded finding reached 9 million by 2077."
    audited = supervisor._audit_report(report)

    assert audited.startswith("# Report")
    assert "## Evidence Graph Validation" in audited
    # With no evidence, every claim is structurally ungrounded.
    assert "0/1 claims structurally grounded" in audited
    assert "(0%)." in audited
