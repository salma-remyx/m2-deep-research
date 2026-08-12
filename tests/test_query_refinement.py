"""Tests for the pre-pipeline query refiner and its supervisor wiring.

The refiner is adapted from G-STEER (arXiv:2608.05876v1). These tests cover
both the refiner in isolation and its integration into the existing
:class:`~src.agents.supervisor.SupervisorAgent` -- the call site -- which is
what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.query_refinement import (
    DEFAULT_FACTORS,
    FactorState,
    FramingFactor,
    QueryRefinementAgent,
    RefinementResult,
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
# Refiner unit tests (new module)
# --------------------------------------------------------------------------- #


def test_objective_always_grounded_for_nonempty_request():
    agent = QueryRefinementAgent()
    result = agent.analyze("quantum computing")

    # A bare topic still states an objective; everything else is under-specified.
    objective = next(f for f in result.factors if f.factor.id == "objective")
    assert objective.grounded is True
    assert result.grounded_count == 1
    assert result.elicited_count == len(DEFAULT_FACTORS) - 1
    assert result.coverage == pytest.approx(1 / len(DEFAULT_FACTORS))


def test_keyword_grounding_marks_specified_factors():
    agent = QueryRefinementAgent()
    # "brief"/"overview" -> depth, "technical"/"engineers" -> audience,
    # "recent"/"trends" -> time horizon, plus objective (always).
    result = agent.analyze(
        "Write a brief technical overview of recent AI trends for engineers"
    )
    grounded_ids = {f.factor.id for f in result.factors if f.grounded}

    assert {"objective", "depth", "audience", "time_horizon"} <= grounded_ids
    assert result.coverage > 0.5


def test_context_grounds_additional_factors():
    agent = QueryRefinementAgent()
    alone = agent.analyze("AI")
    grounded_alone = {f.factor.id for f in alone.factors if f.grounded}

    with_context = agent.analyze("AI", context="brief, for engineers")
    grounded_with = {f.factor.id for f in with_context.factors if f.grounded}

    # The extra context stands in for user memory and grounds more factors.
    assert grounded_alone < grounded_with
    assert "audience" in grounded_with and "depth" in grounded_with


def test_elicited_factors_surface_clarifying_questions():
    agent = QueryRefinementAgent()
    spec = agent.refine("quantum computing")

    # Under-specified factors render their G-STEER clarifying question plus a
    # provisional self-elicited answer.
    assert "_Q:" in spec
    assert "Provisional:" in spec
    assert "Grounded in the request" in spec  # objective kept as-is


def test_refine_leads_with_operative_goal():
    agent = QueryRefinementAgent()
    spec = agent.refine("the impact of remote work on productivity")

    # The downstream supervisor / planning agent reads a clear directive first.
    assert spec.startswith("the impact of remote work on productivity")
    assert "## Research framing" in spec
    assert "arXiv:2608.05876" in spec


def test_dependency_order_resolves_before_dependents():
    # Declared out of order with a dependency chain c -> b -> a; the walk must
    # still emit a before b before c.
    factors = (
        FramingFactor("c", "C", "qC", ("kc",), "dC", ("b",)),
        FramingFactor("b", "B", "qB", ("kb",), "dB", ("a",)),
        FramingFactor("a", "A", "qA", ("ka",), "dA"),
    )
    agent = QueryRefinementAgent(factors)
    order = [f.id for f in agent.graph.in_dependency_order()]

    assert order == ["a", "b", "c"]


def test_refinement_result_counts_are_consistent():
    agent = QueryRefinementAgent()
    result = agent.analyze("compare GPT and Claude")

    assert result.grounded_count + result.elicited_count == len(DEFAULT_FACTORS)
    assert all(isinstance(f, FactorState) for f in result.factors)
    assert isinstance(result, RefinementResult)


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_query_refiner(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.query_refiner, QueryRefinementAgent)


def test_supervisor_refine_query_returns_spec(patch_config_keys):
    supervisor = SupervisorAgent()

    refined = supervisor._refine_query("quantum computing")

    # The helper delegates to the refiner and returns its rendered spec, which
    # is what flows into the pipeline in place of the raw request.
    assert refined.startswith("quantum computing")
    assert "Research framing" in refined
    assert "Intent coverage:" in refined


def test_supervisor_refine_query_enriches_bare_request(patch_config_keys):
    supervisor = SupervisorAgent()
    bare = "robotics"

    refined = supervisor._refine_query(bare)

    # A bare topic is expanded with framing context it did not carry before.
    assert len(refined) > len(bare)
    assert "Audience" in refined  # an under-specified factor, provisionally set


def test_supervisor_refine_query_never_blocks_on_error(patch_config_keys, monkeypatch):
    supervisor = SupervisorAgent()

    def blow_up(*args, **kwargs):
        raise RuntimeError("boom")

    # If the refiner blows up, research must still proceed on the raw request.
    monkeypatch.setattr(supervisor.query_refiner, "analyze", blow_up)

    assert supervisor._refine_query("anything") == "anything"


def test_research_feeds_refined_spec_into_the_pipeline(patch_config_keys, monkeypatch):
    """The top of research() refines the request before it reaches the model."""
    supervisor = SupervisorAgent()

    refined_calls = []
    real_refine = supervisor._refine_query

    def spy(query):
        refined_calls.append(query)
        return real_refine(query)

    monkeypatch.setattr(supervisor, "_refine_query", spy)

    # Stub the streaming call so research() exits at end_turn with no network.
    class _FakeMessage:
        stop_reason = "end_turn"
        content = []

    class _FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter([])

        def get_final_message(self):
            return _FakeMessage()

    monkeypatch.setattr(
        supervisor.client.messages, "stream", lambda **kwargs: _FakeStream()
    )

    supervisor.research("remote work productivity", max_iterations=1)

    # The refiner was invoked with the raw request, and the refined spec --
    # not the raw string -- is what entered the conversation history.
    assert refined_calls == ["remote work productivity"]
    assert "Research framing" in supervisor.messages[0]["content"]
