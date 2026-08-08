"""Tests for the Speculative Actions middleware and its supervisor wiring.

Speculative Actions is adapted from *Speculative Actions: A Lossless
Framework for Faster Agentic Systems* (arXiv:2510.04371v2). These tests
cover the middleware in isolation and its integration into the existing
:class:`~src.agents.supervisor.SupervisorAgent` -- the call site --
which is what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.speculative_actions import SpeculativeActions, WorkflowPredictor
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# SpeculativeActions unit tests (new module)
# --------------------------------------------------------------------------- #


def test_serves_pre_executed_result_on_exact_match():
    calls = []

    def executor(tool_name, tool_input):
        calls.append((tool_name, tool_input["research_query"]))
        return f"findings-for-{tool_input['research_query']}"

    spec = SpeculativeActions(executor)
    # After planning, the predictor speculates the retriever with the
    # planner's output threaded in as subqueries_json.
    spec.prime(
        "planning_agent", {"research_query": "quantum"}, '{"subqueries": [...]}'
    )
    result = spec.execute(
        "web_search_retriever",
        {"research_query": "quantum", "subqueries_json": '{"subqueries": [...]}'},
    )

    assert result == "findings-for-quantum"
    assert spec.stats["hits"] == 1
    assert spec.stats["misses"] == 0
    # The retriever ran exactly once (the speculation) -- never a second time.
    assert calls == [("web_search_retriever", "quantum")]


def test_runs_fresh_on_miss_and_counts_it():
    def executor(tool_name, tool_input):
        return f"fresh-{tool_name}"

    spec = SpeculativeActions(executor)
    spec.prime("planning_agent", {"research_query": "Q"}, "subqueries-blob")
    # Model emits a *different* input than predicted -> miss.
    result = spec.execute(
        "web_search_retriever",
        {"research_query": "Q", "subqueries_json": "something-else"},
    )

    assert result == "fresh-web_search_retriever"
    assert spec.stats["hits"] == 0
    assert spec.stats["misses"] == 1


def test_lossless_served_result_matches_a_fresh_execution():
    """On a hit the served result equals a fresh call for the same input."""
    counter = {"n": 0}

    def executor(tool_name, tool_input):
        counter["n"] += 1
        # Deterministic: identical input -> identical output.
        return (
            f"r:{tool_name}:"
            f"{tool_input.get('research_query')}:{tool_input.get('subqueries_json')}"
        )

    spec = SpeculativeActions(executor)
    spec.prime("planning_agent", {"research_query": "Q"}, "SUBS")
    served = spec.execute(
        "web_search_retriever",
        {"research_query": "Q", "subqueries_json": "SUBS"},
    )
    fresh = executor(
        "web_search_retriever", {"research_query": "Q", "subqueries_json": "SUBS"}
    )

    assert served == fresh  # lossless
    # One speculation + one explicit fresh check; no redundant speculative call.
    assert counter["n"] == 2


def test_predictor_declines_after_retriever():
    pred = WorkflowPredictor()
    # Nothing sensible to speculate once the retriever has run (next is synthesis).
    assert pred.predict_next("web_search_retriever", {}, "findings") is None


def test_no_speculation_served_until_primed():
    def executor(tool_name, tool_input):
        return "plain"

    spec = SpeculativeActions(executor)
    # No prime() call -> plain execution, counted as neither hit nor miss.
    assert spec.execute("planning_agent", {"research_query": "Q"}) == "plain"

    assert spec.stats["hits"] == 0
    assert spec.stats["misses"] == 0


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_speculation_is_off_by_default(patch_config_keys):
    supervisor = SupervisorAgent()

    assert supervisor._speculative is None
    # enable_speculative_actions wires the middleware and returns it.
    assert isinstance(supervisor.enable_speculative_actions(), SpeculativeActions)
    assert supervisor._speculative is not None


def test_supervisor_tool_loop_serves_speculative_retriever(patch_config_keys):
    """Loop wiring hits: planning primes the retriever; the model's matching
    retriever call is served from its pre-execution with no redundant network
    call, and the result + side effects are correct (lossless)."""
    supervisor = SupervisorAgent()
    supervisor.enable_speculative_actions()

    retriever_calls = []

    # Stub the sub-agents so execute_tool does no network I/O.
    supervisor.planning_agent.execute = lambda query: f'{{"subqueries": ["{query}"]}}'

    def fake_retrieve(query, subqueries_json):
        retriever_calls.append((query, subqueries_json))
        return f"FINDINGS({query})"

    supervisor.web_search_retriever.retrieve = fake_retrieve
    # _execute_tool_raw reads last_search_results off the retriever; seed it.
    supervisor.web_search_retriever.last_search_results = [
        {"subquery": "q", "results": []}
    ]

    # --- Iteration 1: the model calls planning_agent. ---
    plan_result = supervisor.execute_tool(
        "planning_agent", {"research_query": "quantum computing"}
    )
    assert plan_result == '{"subqueries": ["quantum computing"]}'
    # The loop then primes the next speculation from this tool's result.
    supervisor._prime_speculation(
        "planning_agent", {"research_query": "quantum computing"}, plan_result
    )
    # The predicted retriever call is now pre-executing on a background thread.
    assert supervisor._speculative.stats["predictions"] == 1

    # --- Iteration 2: the model calls web_search_retriever with the
    # planner's output, exactly as predicted. ---
    findings = supervisor.execute_tool(
        "web_search_retriever",
        {"research_query": "quantum computing", "subqueries_json": plan_result},
    )

    assert findings == "FINDINGS(quantum computing)"
    # Lossless + saved work: the retriever ran exactly once (the speculation),
    # never a second time for the served hit.
    assert retriever_calls == [("quantum computing", plan_result)]
    stats = supervisor._speculative.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    # Side effects of the speculated call are visible to the supervisor.
    assert supervisor._gathered_sources == [{"subquery": "q", "results": []}]


def test_supervisor_loop_miss_runs_real_call_when_unpredicted(patch_config_keys):
    """When the model deviates from the prediction, the loop runs the real
    call fresh and records a miss -- still lossless."""
    supervisor = SupervisorAgent()
    supervisor.enable_speculative_actions()

    supervisor.planning_agent.execute = lambda query: "PLAN"
    supervisor.web_search_retriever.retrieve = lambda q, s: "REAL"
    supervisor.web_search_retriever.last_search_results = []

    supervisor.execute_tool("planning_agent", {"research_query": "Q"})
    supervisor._prime_speculation("planning_agent", {"research_query": "Q"}, "PLAN")
    # Model deviates: different subqueries than the planner produced.
    findings = supervisor.execute_tool(
        "web_search_retriever",
        {"research_query": "Q", "subqueries_json": "DIFFERENT"},
    )

    assert findings == "REAL"
    assert supervisor._speculative.stats["hits"] == 0
    assert supervisor._speculative.stats["misses"] == 1
