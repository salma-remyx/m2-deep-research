"""Tests for the GRADRAG cross-component prompt adapter and its supervisor wiring.

The adapter is adapted from GRADRAG's Prompt Optimizer (arXiv:2607.21324v1).
These tests cover both the adapter in isolation and its integration into the
existing :class:`~src.agents.supervisor.SupervisorAgent` -- the call site --
which is what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.auditor import AuditResult
from src.agents.prompt_adaptation import PromptAdapter, PromptDirective
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring


def _failing_audit() -> AuditResult:
    """An audit with both an ungrounded citation and an unsupported claim."""
    return AuditResult(
        total_citations=3,
        grounded_citations=1,
        unsupported_citations=["https://example.com/fabricated"],
        unsupported_claims=["The Mars colony reached 9 million settlers by 2077."],
        sources_checked=5,
        verifiable=True,
        score=1 / 3,
    )


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# Adapter unit tests (new module)
# --------------------------------------------------------------------------- #


def test_clean_audit_early_stops():
    adapter = PromptAdapter()
    directive = adapter.adapt(AuditResult(), refinement=0)

    assert directive.should_stop is True
    assert directive.needs_refinement is False
    assert "satisfactory" in directive.reason


def test_failing_audit_requests_refinement_with_focus_terms():
    adapter = PromptAdapter()
    directive = adapter.adapt(_failing_audit(), refinement=0)

    assert directive.should_stop is False
    assert directive.needs_refinement is True
    # Focus terms distilled from the unsupported claim drive the next plan.
    assert "mars" in directive.focus_terms
    assert "settlers" in directive.focus_terms
    # The feedback names the gaps so the next turn can act on them.
    assert "1 of 3" in directive.feedback
    assert "mars" in directive.feedback


def test_failing_audit_stops_when_budget_exhausted():
    adapter = PromptAdapter(max_refinements=2)
    directive = adapter.adapt(_failing_audit(), refinement=2)

    assert directive.should_stop is True
    assert "budget exhausted" in directive.reason


def test_unverifiable_audit_stops_with_nothing_to_adapt():
    adapter = PromptAdapter()
    unverifiable = AuditResult(verifiable=False, score=1.0, sources_checked=0)
    directive = adapter.adapt(unverifiable, refinement=0)

    assert directive.should_stop is True
    assert "unverifiable" in directive.reason


def test_planning_advisory_lists_focus_topics():
    adapter = PromptAdapter()
    directive = adapter.adapt(_failing_audit(), refinement=0)
    advisory = adapter.planning_advisory(directive)

    assert "Adaptive focus" in advisory
    assert "mars" in advisory


def test_planning_advisory_empty_when_no_focus():
    adapter = PromptAdapter()
    # Unsupported citation but no unsupported claim -> no focus terms distilled.
    audit = AuditResult(
        total_citations=1,
        grounded_citations=0,
        unsupported_citations=["https://example.com/x"],
        verifiable=True,
        score=0.0,
    )
    directive = adapter.adapt(audit, refinement=0)

    assert directive.focus_terms == []
    assert adapter.planning_advisory(directive) == ""


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_prompt_adapter(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.prompt_adapter, PromptAdapter)
    assert supervisor._refinement_iteration == 0
    assert supervisor._base_planning_prompt == supervisor.planning_agent.system_prompt
    # Fresh state: a clean audit so nothing refines until evidence says otherwise.
    assert supervisor.last_audit_result.passed is True


def test_supervisor_adapts_from_failing_audit(patch_config_keys):
    supervisor = SupervisorAgent()
    supervisor.last_audit_result = _failing_audit()

    directive = supervisor._adapt_from_audit(0)

    assert isinstance(directive, PromptDirective)
    assert directive.should_stop is False
    assert "mars" in directive.focus_terms


def test_supervisor_applies_prompt_adaptation_to_planning_and_turn(patch_config_keys):
    supervisor = SupervisorAgent()
    supervisor.last_audit_result = _failing_audit()
    directive = supervisor._adapt_from_audit(0)

    base_prompt = supervisor.planning_agent.system_prompt
    supervisor._apply_prompt_adaptation(directive)

    # The adaptive planning agent's prompt is extended with the focus topics.
    assert supervisor.planning_agent.system_prompt != base_prompt
    assert "Adaptive focus" in supervisor.planning_agent.system_prompt
    assert "mars" in supervisor.planning_agent.system_prompt
    # The critique is also injected as a supervisor turn so the loop acts on it.
    last = supervisor.messages[-1]
    assert last["role"] == "user"
    assert "Grounding audit feedback" in last["content"]


def test_supervisor_early_stops_on_clean_audit(patch_config_keys):
    supervisor = SupervisorAgent()
    # A clean audit (default) -> early stop, no prompt mutation.
    supervisor.last_audit_result = AuditResult()

    directive = supervisor._adapt_from_audit(0)

    assert directive.should_stop is True


def test_audit_report_exposes_result_for_adapter(patch_config_keys):
    """_audit_report must stash its result so the adapter can read it."""
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = [
        {
            "url": "https://example.com/real",
            "title": "Real",
            "text": "real evidence here",
            "highlights": [],
        }
    ]
    report = "Backed by [the source](https://example.com/real) and [fake](https://x.io/nope)."

    supervisor._audit_report(report)

    exposed = supervisor.last_audit_result
    assert exposed.total_citations == 2
    assert exposed.grounded_citations == 1
    assert "https://x.io/nope" in exposed.unsupported_citations
