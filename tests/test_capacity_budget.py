"""Tests for the per-role capacity budget and its supervisor wiring.

The budget is adapted from the capacity-distribution analysis in *Think Big,
Search Small: Where Capacity Matters in Hierarchical Search Agents?*
(arXiv:2607.07548v1). These tests cover the budget in isolation and its
integration into the existing :class:`~src.agents.supervisor.SupervisorAgent`
-- the call site -- which is what proves the wiring actually invokes the new
code.
"""

from types import SimpleNamespace

import pytest

from src.agents.capacity_budget import (
    DELEGATION,
    EXECUTION,
    CapacityBudget,
)
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# CapacityBudget unit tests (new module)
# --------------------------------------------------------------------------- #


def _budget_with_roles() -> CapacityBudget:
    budget = CapacityBudget()
    budget.register(DELEGATION, "MiniMax-M2.1")
    budget.register(EXECUTION, "google/gemini-2.5-flash")
    return budget


def test_records_anthropic_and_openrouter_usage():
    """Delegation usage is an Anthropic Usage object; execution is an
    OpenRouter dict -- both must be normalized into per-role token totals."""
    budget = _budget_with_roles()

    # Anthropic-style usage object (attributes).
    budget.record_delegation(SimpleNamespace(input_tokens=200, output_tokens=80))
    # OpenRouter-style usage dict (prompt/completion keys).
    budget.record_execution({"prompt_tokens": 100, "completion_tokens": 50})

    dist = budget.distribution()
    assert dist.delegation.total_tokens == 280
    assert dist.execution.total_tokens == 150
    assert dist.total_tokens == 430
    assert dist.delegation_share == pytest.approx(280 / 430)
    assert dist.execution_share == pytest.approx(150 / 430)
    # Delegation out-spends execution -> matches the 'think big' structure.
    assert dist.matches_recipe is True


def test_execution_outspending_delegation_violates_recipe():
    budget = _budget_with_roles()
    budget.record_delegation(SimpleNamespace(input_tokens=10, output_tokens=10))
    budget.record_execution({"prompt_tokens": 500, "completion_tokens": 500})

    assert budget.distribution().matches_recipe is False


def test_recommend_downsize_when_execution_measured():
    budget = _budget_with_roles()
    budget.record_execution({"prompt_tokens": 1000, "completion_tokens": 200})

    rec = budget.recommend_downsize()

    assert rec.downsize_execution is True
    assert rec.execution_output_tokens == 200
    # Paper's Pareto anchor: 0.37 * measured execution output tokens.
    assert rec.estimated_token_saving == round(200 * 0.37)
    assert rec.paper_reference == "arXiv:2607.07548v1"
    assert "11" in rec.rationale and "2.6" in rec.rationale


def test_no_recommendation_when_execution_unmeasured():
    budget = _budget_with_roles()
    # Only delegation spend recorded this run.
    budget.record_delegation(SimpleNamespace(input_tokens=500, output_tokens=200))

    rec = budget.recommend_downsize()

    assert rec.downsize_execution is False
    assert rec.estimated_token_saving == 0


def test_none_usage_is_safe():
    budget = _budget_with_roles()
    budget.record_delegation(None)
    budget.record_execution(None)

    assert budget.distribution().total_tokens == 0


def test_reset_clears_usage_but_keeps_registration():
    budget = _budget_with_roles()
    budget.record_delegation(SimpleNamespace(input_tokens=100, output_tokens=40))

    budget.reset()

    dist = budget.distribution()
    assert dist.total_tokens == 0
    assert dist.delegation.calls == 0
    # Role -> model registration survives a reset.
    assert dist.delegation.model == "MiniMax-M2.1"
    assert dist.execution.model == "google/gemini-2.5-flash"


def test_render_contains_section_and_attribution():
    budget = _budget_with_roles()
    budget.record_delegation(SimpleNamespace(input_tokens=200, output_tokens=80))
    budget.record_execution({"prompt_tokens": 100, "completion_tokens": 50})

    rendered = budget.render()

    assert "## Capacity Budget" in rendered
    assert "arXiv:2607.07548v1" in rendered
    assert "MiniMax-M2.1" in rendered
    assert "google/gemini-2.5-flash" in rendered
    assert "Estimated saving" in rendered  # execution was measured


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_capacity_budget(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.capacity, CapacityBudget)
    dist = supervisor.capacity.distribution()
    # Roles are registered against the configured models at construction.
    assert dist.delegation.model == "MiniMax-M2.1"
    assert dist.execution.model == "google/gemini-2.5-flash"
    assert dist.total_tokens == 0


def test_supervisor_records_execution_usage_in_tool_loop(patch_config_keys):
    """Drive the supervisor's planning-agent tool path and assert the capacity
    budget picks up the execution call's usage -- reproducing what
    execute_tool does on a tool turn, with no network call."""
    supervisor = SupervisorAgent()

    # Stub the planning agent so no network call is made, and expose usage.
    supervisor.planning_agent.execute = lambda query: '{"subqueries": []}'
    supervisor.planning_agent.last_usage = {
        "prompt_tokens": 120,
        "completion_tokens": 60,
    }

    out = supervisor.execute_tool("planning_agent", {"research_query": "q"})

    assert out == '{"subqueries": []}'
    execution = supervisor.capacity.distribution().execution
    assert execution.input_tokens == 120
    assert execution.output_tokens == 60
    assert execution.calls == 1


def test_supervisor_capacity_section_appends_to_report(patch_config_keys):
    supervisor = SupervisorAgent()
    # Simulate usage captured during a run.
    supervisor.capacity.record_delegation(
        SimpleNamespace(input_tokens=300, output_tokens=120)
    )
    supervisor.capacity.record_execution(
        {"prompt_tokens": 150, "completion_tokens": 90}
    )

    combined = supervisor._append_capacity_report("# Report\n\nFindings.")

    # Original report is preserved and the capacity section is appended.
    assert combined.startswith("# Report")
    assert "## Capacity Budget" in combined
    assert "Downsizing recommendation" in combined
