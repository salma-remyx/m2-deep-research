"""Tests for the DeepStress evidence-robustness probe and its supervisor wiring.

The stressor is adapted from *DeepStress: Stress-Testing Deep Search Agents*
(arXiv:2607.13920v1). These tests exercise the stressor against the existing
:class:`~src.agents.auditor.ReportAuditor` -- the non-new module the probe is
designed to stress -- and its opt-in hook on
:class:`~src.agents.supervisor.SupervisorAgent` (the call site), which proves
the wiring actually invokes the new code.
"""

import pytest

from src.agents.auditor import ReportAuditor  # non-new module -> the component under stress
from src.agents.evidence_stressor import (
    DIMENSIONS,
    FACTUALITY,
    RELEVANCE,
    TRUSTWORTHINESS,
    EvidenceStressor,
)
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# Three clean sources (mirror ExaTool.format_results output). Three so a
# challenge_rate of 1.0 yields one challenge per DeepStress dimension.
CLEAN_SOURCES = [
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
    {
        "title": "Quantum Investment Trends",
        "url": "https://example.com/quantum-investment",
        "text": "Venture funding for quantum startups reached 1.2 billion in 2024.",
        "highlights": ["venture funding 1.2 billion"],
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
# Stressor x auditor (exercises the non-new ReportAuditor)
# --------------------------------------------------------------------------- #


def test_inject_controls_challenge_rate_across_all_dimensions():
    stressor = EvidenceStressor()
    stressed, injected = stressor.inject(
        CLEAN_SOURCES, "quantum computing", challenge_rate=1.0
    )

    # One challenge per clean source at rate 1.0, appended to the clean set.
    assert len(injected) == len(CLEAN_SOURCES)
    assert len(stressed) == len(CLEAN_SOURCES) + len(injected)
    # Round-robin over the three dimensions covers all of DeepStress's axes.
    assert {c.dimension for c in injected} == set(DIMENSIONS)


def test_inject_zero_rate_adds_nothing():
    stressor = EvidenceStressor()
    stressed, injected = stressor.inject(
        CLEAN_SOURCES, "quantum computing", challenge_rate=0.0
    )
    assert injected == []
    assert stressed == CLEAN_SOURCES


def test_probe_detects_trustworthiness_and_relevance():
    """Fabricated URLs and off-topic claims are caught by the parameter-free auditor."""
    stressor = EvidenceStressor()
    auditor = ReportAuditor()
    report = stressor.probe(auditor, CLEAN_SOURCES, "quantum computing", challenge_rate=1.0)

    by_dim = report.detection_by_dimension()
    assert by_dim[TRUSTWORTHINESS] == 1.0
    assert by_dim[RELEVANCE] == 1.0


def test_probe_cannot_detect_factuality_conflicts():
    """Honest blind spot: a lexical-overlap auditor grounds a claim that reuses a
    clean source's words with one number flipped, so factuality conflicts slip
    through. This is exactly the parametric-vs-retrieved conflict DeepStress
    highlights -- the probe documents it rather than hiding it."""
    stressor = EvidenceStressor()
    auditor = ReportAuditor()
    report = stressor.probe(auditor, CLEAN_SOURCES, "quantum computing", challenge_rate=1.0)

    by_dim = report.detection_by_dimension()
    assert by_dim[FACTUALITY] == 0.0
    # Trustworthiness + relevance caught, factuality missed -> partial detection.
    assert 0.0 < report.detection_rate < 1.0


def test_probe_is_deterministic_with_seed():
    stressor = EvidenceStressor()
    auditor = ReportAuditor()
    a = stressor.probe(auditor, CLEAN_SOURCES, "quantum computing", challenge_rate=1.0, seed=7)
    b = stressor.probe(auditor, CLEAN_SOURCES, "quantum computing", challenge_rate=1.0, seed=7)

    assert [o.detected for o in a.outcomes] == [o.detected for o in b.outcomes]
    assert a.challenge_rate_actual == b.challenge_rate_actual


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_exposes_stress_probe(patch_config_keys):
    supervisor = SupervisorAgent()
    assert isinstance(supervisor.evidence_stressor, EvidenceStressor)


def test_supervisor_stress_test_grounding_runs_over_gathered_sources(patch_config_keys):
    """Call site: SupervisorAgent.stress_test_grounding() invokes the new
    EvidenceStressor against the auditor over the run's own gathered sources."""
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = CLEAN_SOURCES

    report = supervisor.stress_test_grounding("quantum computing", challenge_rate=1.0)

    assert report.n_clean == len(CLEAN_SOURCES)
    assert report.n_injected == len(CLEAN_SOURCES)
    # Same honest detection matrix as the direct probe.
    by_dim = report.detection_by_dimension()
    assert by_dim[TRUSTWORTHINESS] == 1.0
    assert by_dim[RELEVANCE] == 1.0
    assert by_dim[FACTUALITY] == 0.0
