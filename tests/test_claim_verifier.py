"""Tests for the AgentKGV-style claim re-verifier and its supervisor wiring.

The re-verifier adapts the iterative query-rewriting retrieval mechanism from
*AgentKGV* (arXiv:2607.09092v1). These tests cover the verifier in isolation
(driven by a fake Exa so no network call is made) and its integration into the
existing :class:`~src.agents.supervisor.SupervisorAgent` -- the call site --
which is what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.claim_verifier import ClaimVerification, ClaimVerifier, VerificationReport
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# A claim the auditor will flag: numbers + no overlap with the quantum sources.
MARS_CLAIM = "The Mars colony population reached 9 million settlers by 2077."

# Sources about an unrelated topic, so the Mars claim is unsupported.
QUANTUM_SOURCES = [
    {
        "title": "Quantum Computing Market Report",
        "url": "https://example.com/quantum-report",
        "text": "The quantum computing market is projected to reach 47 billion "
        "dollars by 2030 according to industry analysts.",
        "highlights": ["market projected to reach 47 billion"],
    }
]


class _FakeExa:
    """Stand-in for ExaTool that returns canned sources in call order.

    The verifier calls ``exa.search`` then ``exa.format_results`` once per
    rewrite attempt, so mapping responses to call order lets a test pin which
    rewrite grounded a claim without coupling to exact query strings.
    """

    def __init__(self, result_sequence):
        self._results = list(result_sequence)
        self._index = 0
        self.queries = []

    def search(self, query, num_results=5, **kwargs):
        self.queries.append(query)
        return {"_q": query}

    def format_results(self, results):
        out = self._results[self._index] if self._index < len(self._results) else []
        self._index += 1
        return out


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# ClaimVerifier unit tests (new module)
# --------------------------------------------------------------------------- #


def test_recovers_claim_on_first_rewrite():
    ground = [
        {
            "url": "https://example.com/mars",
            "title": "Mars Colony",
            "text": "The Mars colony population reached 9 million settlers by 2077.",
            "highlights": [],
        }
    ]
    verifier = ClaimVerifier(exa=_FakeExa([ground]))

    report = verifier.verify([MARS_CLAIM])

    assert report.total == 1
    assert report.verified_count == 1
    assert report.results[0].verified is True
    assert report.results[0].rewrites_attempted == 1
    assert report.results[0].supporting_url == "https://example.com/mars"


def test_recovers_claim_on_a_later_rewrite():
    # First rewrite returns an unrelated source (no overlap); the second returns
    # a grounding one. Proves the loop iterates instead of stopping at attempt 1.
    unrelated = [{"url": "https://example.com/noise", "title": "Noise",
                  "text": "completely unrelated ocean fishing weather report", "highlights": []}]
    ground = [{"url": "https://example.com/mars", "title": "Mars Colony",
               "text": "Mars colony population 9 million settlers 2077", "highlights": []}]
    verifier = ClaimVerifier(exa=_FakeExa([unrelated, ground]))

    report = verifier.verify([MARS_CLAIM])

    assert report.results[0].verified is True
    assert report.results[0].rewrites_attempted == 2
    assert report.results[0].supporting_url == "https://example.com/mars"


def test_misses_claim_when_no_rewrite_grounds():
    # Every rewrite returns sources with no term overlap -> still unsupported.
    noise = [{"url": "https://example.com/x", "title": "X",
              "text": "ocean fishing weather report unrelated", "highlights": []}]
    verifier = ClaimVerifier(exa=_FakeExa([noise, noise, noise]))

    report = verifier.verify([MARS_CLAIM])

    assert report.verified_count == 0
    assert report.results[0].verified is False
    assert report.results[0].supporting_url is None
    assert report.results[0].rewrites_attempted >= 1


def test_first_rewrite_strips_stopwords_into_key_terms():
    # The verifier's parameter-free rewriter: first query is the stopword-
    # stripped key-term surface form of the claim, not the raw sentence.
    fake = _FakeExa([[]])
    verifier = ClaimVerifier(exa=fake, max_rewrites=1)

    verifier.verify([MARS_CLAIM])

    first_query = fake.queries[0]
    assert "the" not in first_query.split()
    assert "mars" in first_query
    assert "colony" in first_query


def test_verify_caps_claims_at_max_claims():
    fake = _FakeExa([])  # returns nothing -> every claim misses fast
    verifier = ClaimVerifier(exa=fake, max_claims=2, max_rewrites=1)

    report = verifier.verify(["claim one 123", "claim two 456", "claim three 789"])

    assert report.total == 2


def test_format_section_marks_recovered_and_still_unsupported():
    results = [
        ClaimVerification(claim="recovered claim 100", verified=True,
                          query_used="q", supporting_url="https://example.com/a",
                          rewrites_attempted=2),
        ClaimVerification(claim="missed claim 200", verified=False,
                          query_used=None, supporting_url=None, rewrites_attempted=3),
    ]
    verifier = ClaimVerifier(exa=_FakeExa([]))

    rendered = verifier.format_section(VerificationReport(results=results))

    assert "iterative query rewriting" in rendered
    assert "arXiv:2607.09092" in rendered
    assert "Recovered" in rendered
    assert "Still unsupported" in rendered
    assert "https://example.com/a" in rendered


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_claim_verifier(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.claim_verifier, ClaimVerifier)


def test_supervisor_audit_invokes_verifier_on_unsupported_claims(patch_config_keys):
    supervisor = SupervisorAgent()
    # Quantum sources -> the Mars claim is unsupported, triggering the verifier.
    supervisor._gathered_sources = QUANTUM_SOURCES
    # Inject a fake Exa so the verifier recovers the Mars claim with no network.
    supervisor.claim_verifier.exa = _FakeExa(
        [[{
            "url": "https://example.com/mars",
            "title": "Mars Colony",
            "text": "The Mars colony population reached 9 million settlers by 2077.",
            "highlights": [],
        }]]
    )

    audited = supervisor._audit_report(MARS_CLAIM)

    # Auditor section still present, and the re-verification section was appended.
    assert "Source Grounding Audit" in audited
    assert "iterative query rewriting" in audited
    assert "Recovered" in audited
    assert "https://example.com/mars" in audited


def test_supervisor_skips_verifier_when_report_is_clean(patch_config_keys):
    supervisor = SupervisorAgent()
    # A fully grounded report -> no unsupported claims -> verifier must not run.
    supervisor._gathered_sources = [
        {
            "url": "https://example.com/quantum-report",
            "title": "Quantum Report",
            "text": "The quantum computing market is projected to reach 47 "
            "billion by 2030.",
            "highlights": [],
        }
    ]
    # If the verifier ran it would hit this fake; assert the section is absent.
    supervisor.claim_verifier.exa = _FakeExa([{"url": "should-not-run"}])

    report = (
        "The quantum computing market is projected to reach 47 billion by 2030 "
        "[report](https://example.com/quantum-report)."
    )
    audited = supervisor._audit_report(report)

    assert "Source Grounding Audit" in audited
    assert "iterative query rewriting" not in audited
