"""Tests for the EMAS-style prompt evolution loop and its supervisor wiring.

Adapted from *EMAS: Stabilizing Multi-Agent System Evolution through
Evidence-Guided Revision* (arXiv:2608.07196v1). These tests cover both the
evolution loop in isolation and its integration into the existing
:class:`~src.agents.supervisor.SupervisorAgent` -- the call site -- which is
what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.auditor import ReportAuditor
from src.agents.prompt_evolution import PromptEvolution, Revision
from src.agents.research_trace import ResearchTrace
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# Flat source fixtures (mirror ExaTool.format_results output).
SOURCES = [
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


def _bad_report():
    """A report with a fabricated citation and an unsupported numeric claim."""
    return (
        "The Mars colony population reached 9 million settlers by 2077 "
        "[report](https://example.com/made-up).\n"
    )


# --------------------------------------------------------------------------- #
# Unit: the evolution loop
# --------------------------------------------------------------------------- #


def test_no_revision_proposed_before_recurrence():
    """EMAS's recurrence gate: one bad run is noise, not a diagnosis."""
    evo = PromptEvolution(min_recurrence=2)
    audit = ReportAuditor().audit(_bad_report(), SOURCES)

    assert evo.diagnose(audit) == ["fabricated_citations", "ungrounded_claims"]
    assert evo.observe_run(audit) is None
    assert evo.pending() == []


def test_recurring_diagnosis_proposes_one_revision():
    evo = PromptEvolution(min_recurrence=2)
    audit = ReportAuditor().audit(_bad_report(), SOURCES)

    assert evo.observe_run(audit) is None  # first occurrence: evidence only
    proposed = evo.observe_run(audit)  # second occurrence: revise

    assert isinstance(proposed, Revision)
    assert proposed.status == "candidate"
    assert proposed.target == "planning_agent"
    assert proposed.evidence == 2
    assert len(evo.state.revisions) == 1


def test_clean_run_accumulates_nothing():
    evo = PromptEvolution()
    audit = ReportAuditor().audit(
        "Backed by [the report](https://example.com/quantum-report).", SOURCES
    )

    assert evo.diagnose(audit) == []
    evo.observe_run(audit)

    assert evo.state.diagnosis_counts == {}


def test_unverifiable_audit_is_not_a_diagnosis():
    """No sources -> unverifiable -> not penalized, matching the auditor."""
    evo = PromptEvolution()
    audit = ReportAuditor().audit(_bad_report(), [])

    assert evo.diagnose(audit) == []


def test_thin_evidence_diagnosed_from_trace():
    evo = PromptEvolution()
    trace = ResearchTrace()
    trace.record_subgoal("q")
    trace.record_tool("web_search_retriever", {"research_query": "q"})
    trace.record_evidence([{"url": "https://example.com/a"}])

    assert evo.diagnose(None, trace) == ["thin_evidence"]


def test_candidate_accepted_when_score_does_not_drop():
    evo = PromptEvolution(min_recurrence=2)
    audit = ReportAuditor().audit(_bad_report(), SOURCES)
    evo.observe_run(audit)
    proposed = evo.observe_run(audit)

    # First revised run sets the baseline and accepts.
    evo.validate(0.9)
    assert proposed.status == "accepted"

    # A later drop below the baseline rejects a pending candidate.
    evo.state.revisions.append(
        Revision(diagnosis="thin_evidence", target="planning_agent", instruction="x")
    )
    evo.validate(0.5)
    assert evo.state.revisions[-1].status == "rejected"


def test_apply_accepted_targets_the_right_agent():
    evo = PromptEvolution(min_recurrence=1)
    audit = ReportAuditor().audit(_bad_report(), SOURCES)
    evo.observe_run(audit)
    evo.validate(0.9)

    planning = evo.apply_accepted("PLAN", "planning_agent")
    retriever = evo.apply_accepted("RETRIEVE", "web_search_retriever")

    assert planning.startswith("PLAN") and len(planning) > len("PLAN")
    assert retriever == "RETRIEVE"
    # Idempotent: applying twice does not duplicate the instruction.
    assert evo.apply_accepted(planning, "planning_agent") == planning


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_evolution(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.prompt_evolution, PromptEvolution)
    assert supervisor._last_audit is None


def test_supervisor_evolution_learns_from_recurring_failures(patch_config_keys):
    """Two failing runs through the supervisor propose a revision; the next
    run's prompts carry it (the EMAS loop wired into the real run flow)."""
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = SOURCES

    for _ in range(2):
        supervisor._audit_report(_bad_report())
        supervisor._evolve_from_run()

    assert supervisor.prompt_evolution.pending() != []
    original_prompt = supervisor.planning_agent.system_prompt

    supervisor._apply_evolved_prompts()

    assert supervisor.planning_agent.system_prompt != original_prompt
    assert supervisor.planning_agent.system_prompt.startswith(original_prompt)
    # The candidate is applied so the revised system actually runs before
    # validation judges it.
    assert "only cite URLs" in supervisor.planning_agent.system_prompt


def test_supervisor_clean_run_leaves_prompts_untouched(patch_config_keys):
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = SOURCES
    original_prompt = supervisor.planning_agent.system_prompt

    supervisor._audit_report(
        "Backed by [the report](https://example.com/quantum-report)."
    )
    supervisor._evolve_from_run()
    supervisor._apply_evolved_prompts()

    assert supervisor.planning_agent.system_prompt == original_prompt


def test_evolution_state_persists_across_instances(tmp_path):
    path = str(tmp_path / "evolution.json")
    first = PromptEvolution(state_path=path, min_recurrence=1)
    audit = ReportAuditor().audit(_bad_report(), SOURCES)
    first.observe_run(audit)
    first.validate(0.9)

    second = PromptEvolution(state_path=path)

    assert second.state.diagnosis_counts == first.state.diagnosis_counts
    assert [r.status for r in second.state.revisions] == ["accepted"]
    assert "only cite URLs" in second.apply_accepted("PLAN", "planning_agent")
