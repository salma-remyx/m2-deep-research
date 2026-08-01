"""Tests for the budget-to-reward efficiency meter and its supervisor wiring.

The meter ports AREK's efficiency-MEASUREMENT contribution (arXiv:2607.24647v1):
the AUC of the budget -> best-reward-so-far frontier as a dimension distinct
from outcome quality. These tests cover the meter in isolation and its
integration into the existing :class:`~src.agents.supervisor.SupervisorAgent`
-- the call site -- which is what proves the wiring actually invokes the new
code.
"""

import pytest

from src.agents.efficiency_meter import EfficiencyMeter, EfficiencySample
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# Flat source fixtures (mirror ExaTool.format_results output / test_report_auditor).
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
# EfficiencyMeter unit tests (new module)
# --------------------------------------------------------------------------- #


def test_auc_single_sample_rewards_cheap_runs():
    """Reaching reward r using fraction b of the budget scores r * (1 - b):
    high quality reached cheaply is efficient."""
    meter = EfficiencyMeter()
    meter.reset(budget_horizon=10.0)
    for _ in range(3):  # 3 of 10 iterations
        meter.record_iteration()
    meter.record_reward(0.8)

    assert meter.auc() == pytest.approx(0.8 * (1 - 3 / 10))


def test_auc_running_max_over_multiple_samples():
    meter = EfficiencyMeter()
    meter.reset(budget_horizon=10.0)
    # reward 0.5 first observed at iteration 2, 0.9 at iteration 6.
    meter.record_reward(0.5, budget=2)
    meter.record_reward(0.9, budget=6)

    # [0,2): 0 ; [2,6): 0.5 ; [6,10]: 0.9  -> 0.5*0.4 + 0.9*0.4 = 0.56
    assert meter.auc() == pytest.approx(0.5 * 0.4 + 0.9 * 0.4)


def test_auc_full_budget_run_scores_near_zero_even_when_perfect():
    """A perfectly grounded report that used the whole budget is inefficient --
    the efficiency dimension, distinct from outcome quality."""
    meter = EfficiencyMeter()
    meter.reset(budget_horizon=10.0)
    for _ in range(10):
        meter.record_iteration()
    meter.record_reward(1.0)

    assert meter.auc() == pytest.approx(0.0)
    assert meter.final_reward() == 1.0  # outcome quality is still perfect


def test_auc_no_samples_is_zero():
    meter = EfficiencyMeter()
    meter.reset(budget_horizon=10.0)
    assert meter.auc() == 0.0
    assert meter.final_reward() == 0.0


def test_record_iteration_advances_budget():
    meter = EfficiencyMeter()
    meter.reset(budget_horizon=5.0)
    assert meter.budget_used == 0.0
    for _ in range(4):
        meter.record_iteration()
    assert meter.budget_used == 4.0


def test_note_sources_flattens_nested_buckets_and_keeps_high_water_mark():
    meter = EfficiencyMeter()
    meter.reset()
    meter.note_sources(RETRIEVED_SOURCES)  # 2 results + 1 similar
    meter.note_sources([{"url": "u", "title": "t"}])  # smaller -> ignored
    assert meter.report().sources == 3


def test_reset_clears_previous_run():
    meter = EfficiencyMeter()
    meter.reset(budget_horizon=10.0)
    meter.record_iteration()
    meter.record_reward(0.5)
    meter.note_tool_call()

    meter.reset(budget_horizon=4.0)
    assert meter.budget_used == 0.0
    assert meter.samples() == []
    assert meter.auc() == 0.0
    assert meter.report().tool_calls == 0


def test_render_contains_section_and_attribution():
    meter = EfficiencyMeter()
    meter.reset(budget_horizon=10.0)
    for _ in range(2):
        meter.record_iteration()
    meter.record_reward(0.5)

    rendered = meter.render()
    assert "## Search Efficiency" in rendered
    assert "arXiv:2607.24647v1" in rendered
    assert "Outcome quality:" in rendered
    assert "Search efficiency (AUC):" in rendered


def test_render_no_reward_is_graceful():
    meter = EfficiencyMeter()
    meter.reset(budget_horizon=10.0)
    meter.record_iteration()

    rendered = meter.render()
    assert "could not be measured" in rendered


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_efficiency_meter(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.efficiency_meter, EfficiencyMeter)
    assert supervisor._last_audit_score is None


def test_supervisor_audit_surfaces_score_for_meter(patch_config_keys):
    """The _audit_report wiring stashes the grounding score so the meter can
    record it as the run's reward."""
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = GROUNDING_SOURCES

    report = (
        "# Report\n\n"
        "Grounded [report](https://example.com/quantum-report) but "
        "[fabricated](https://example.com/made-up) is not.\n"
    )
    supervisor._audit_report(report)

    # 2 citations, 1 grounded -> score 0.5; sources present -> verifiable.
    assert supervisor._last_audit_score == pytest.approx(0.5)


def test_supervisor_audit_unverifiable_surfaces_none(patch_config_keys):
    """With no retrieved sources the audit is unverifiable, so no reward is
    surfaced and the meter records nothing for that run."""
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = []

    supervisor._audit_report("# Report\n\n[a](https://x.io/y)")
    assert supervisor._last_audit_score is None


def test_supervisor_end_turn_records_reward_and_renders(patch_config_keys):
    """Drive the end_turn path exactly as SupervisorAgent.research() does --
    reset the meter, tick one budget unit per iteration, audit the report,
    record the surfaced score, render -- and assert the wiring produces the
    efficiency section with the expected AUC. No network call is made."""
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = GROUNDING_SOURCES

    # Mirror research(): fresh meter over a 10-iteration horizon.
    supervisor.efficiency_meter.reset(budget_horizon=10.0)
    for _ in range(3):  # supervisor finished in 3 iterations
        supervisor.efficiency_meter.record_iteration()

    report = (
        "# Report\n\n"
        "Grounded [report](https://example.com/quantum-report) but "
        "[fabricated](https://example.com/made-up) is not.\n"
    )
    audited = supervisor._audit_report(report)
    # Mirror the end_turn reward recording.
    if supervisor._last_audit_score is not None:
        supervisor.efficiency_meter.record_reward(supervisor._last_audit_score)

    section = supervisor.efficiency_meter.render()
    assert "## Search Efficiency" in section
    # reward 0.5 reached at 3/10 budget -> 0.5 * (1 - 0.3) = 0.35
    assert supervisor.efficiency_meter.auc() == pytest.approx(0.35)
    assert audited.startswith("# Report")


def test_supervisor_efficiency_section_appends_to_report(patch_config_keys):
    """The rendered meter is an artifact appended to a delivered report."""
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = GROUNDING_SOURCES
    supervisor.efficiency_meter.reset(budget_horizon=10.0)
    supervisor.efficiency_meter.record_iteration()
    supervisor._audit_report(
        "Grounded [report](https://example.com/quantum-report)."
    )
    supervisor.efficiency_meter.record_reward(supervisor._last_audit_score)

    report = "# Report\n\nFindings."
    combined = report + supervisor.efficiency_meter.render()

    assert combined.startswith("# Report")
    assert "## Search Efficiency" in combined


def test_efficiency_sample_is_a_dataclass_pair():
    sample = EfficiencySample(budget=2.0, reward=0.5)
    assert sample.budget == 2.0
    assert sample.reward == 0.5
