"""Tests for the Structured Evidence Ledger and its auditor wiring.

The ledger is adapted from LedgerMind's Structured Evidence Ledger
(arXiv:2607.28374v1) -- a Mode 2 adapted port where the learned grounding
judge is replaced by a parameter-free lexical-overlap proxy.

The integration tests import the *existing* :class:`~src.agents.auditor.ReportAuditor`
and :class:`~src.agents.supervisor.SupervisorAgent` (non-new modules) and assert
that ``audit()`` now links each claim to its backing ledger entry as provenance
-- which is what proves the wiring actually invokes the new ledger.
"""

import pytest

from src.agents.auditor import ReportAuditor  # non-new module -> proves wiring
from src.agents.evidence_ledger import EvidenceLedger, LedgerEntry
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
# EvidenceLedger unit tests (new module)
# --------------------------------------------------------------------------- #


def test_ledger_assigns_stable_ids():
    ledger = EvidenceLedger(GROUNDING_SOURCES)

    assert ledger.size == 2
    assert [e.entry_id for e in ledger] == ["E1", "E2"]
    assert all(isinstance(e, LedgerEntry) for e in ledger)


def test_find_url_resolves_citation_to_entry():
    ledger = EvidenceLedger(GROUNDING_SOURCES)

    # Normalization: clean https citation matches the http+www+slash source.
    assert ledger.find_url("https://example.com/quantum-report").entry_id == "E1"
    assert ledger.find_url("https://example.com/qubit-fidelity").entry_id == "E2"
    # A URL never retrieved resolves to nothing (no backing entry).
    assert ledger.find_url("https://example.com/fabricated") is None


def test_backing_entries_link_claim_to_specific_evidence():
    ledger = EvidenceLedger(GROUNDING_SOURCES)

    claim = "The quantum computing market is projected to reach 47 billion by 2030."
    backing = ledger.backing_entries(claim)

    # The claim is linked to the *specific* entry that backs it, not the corpus.
    assert [e.entry_id for e in backing] == ["E1"]
    assert backing[0].title == "Quantum Computing Market Report"


def test_backing_entries_empty_for_unsupported_claim():
    ledger = EvidenceLedger(GROUNDING_SOURCES)

    backing = ledger.backing_entries(
        "The Mars colony population reached 9 million settlers by 2077."
    )
    assert backing == []  # no ledger entry backs it -> unsupported


def test_active_entry_ids_are_only_the_cited_subset():
    ledger = EvidenceLedger(GROUNDING_SOURCES)

    cited = [
        "https://example.com/quantum-report",  # gathered -> active
        "https://example.com/fabricated",  # never gathered -> not active
    ]
    assert ledger.active_entry_ids(cited) == {"E1"}


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing ReportAuditor
# --------------------------------------------------------------------------- #


def test_audit_populates_claim_provenance():
    """audit() links each grounded claim to its backing ledger entry(ies)."""
    auditor = ReportAuditor()
    report = (
        "The quantum computing market is projected to reach 47 billion by 2030 "
        "[report](https://example.com/quantum-report)."
    )
    result = auditor.audit(report, GROUNDING_SOURCES)

    assert result.grounded_claims == 1
    assert result.ledger_size == 2
    assert result.cited_entry_ids == ["E1"]
    assert len(result.claim_provenance) == 1
    claim_text, entries = result.claim_provenance[0]
    assert "quantum computing market" in claim_text.lower()
    assert entries == [("E1", "Quantum Computing Market Report")]
    # Existing grounding signals are preserved alongside the new provenance.
    assert result.unsupported_claims == []
    assert result.unsupported_citations == []


def test_audit_provenance_empty_when_claim_unsupported():
    auditor = ReportAuditor()
    report = "The Mars colony population reached 9 million settlers by 2077."
    result = auditor.audit(report, GROUNDING_SOURCES)

    assert result.grounded_claims == 0
    assert result.claim_provenance == []
    assert any("Mars" in claim for claim in result.unsupported_claims)


def test_format_report_renders_provenance_links():
    auditor = ReportAuditor()
    report = (
        "The quantum computing market is projected to reach 47 billion by 2030 "
        "[report](https://example.com/quantum-report)."
    )
    result = auditor.audit(report, GROUNDING_SOURCES)
    rendered = auditor.format_report(result)

    assert "Claim provenance" in rendered
    assert "Structured Evidence Ledger" in rendered
    assert "arXiv:2607.28374" in rendered
    # The claim is visibly linked to its named backing source.
    assert "E1 (Quantum Computing Market Report)" in rendered
    assert "backed by" in rendered


def test_format_report_shows_ledger_utilization():
    auditor = ReportAuditor()
    # Two gathered sources, but only one is cited in the report.
    report = (
        "Quantum growth is real [report](https://example.com/quantum-report)."
    )
    result = auditor.audit(report, GROUNDING_SOURCES)
    rendered = auditor.format_report(result)

    assert "2 gathered source(s)" in rendered
    assert "1 cited in the report" in rendered


# --------------------------------------------------------------------------- #
# Integration: the supervisor's _audit_report call site renders provenance
# --------------------------------------------------------------------------- #


def test_supervisor_audit_renders_ledger_provenance(patch_config_keys):
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = GROUNDING_SOURCES

    report = (
        "# Report\n\n"
        "The quantum computing market is projected to reach 47 billion by 2030 "
        "[report](https://example.com/quantum-report).\n"
    )
    audited = supervisor._audit_report(report)

    assert audited.startswith("# Report")
    assert "Claim provenance" in audited
    assert "backed by E1 (Quantum Computing Market Report)" in audited
