"""Tests for the evidence-state closure tracker and its supervisor wiring.

The tracker is adapted from Omni-Decision's evidence-state system
(arXiv:2607.11433v1). These tests cover the tracker in isolation and its
integration into the existing :class:`~src.agents.supervisor.SupervisorAgent`
-- the call site -- which is what proves the wiring actually invokes the new
code.
"""

import json

import pytest

from src.agents.evidence_state import (
    CONFIRMED,
    CONFLICTING,
    OPEN,
    EvidenceStateTracker,
)
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring


def _bucket(subquery, sources):
    """Build a WebSearchRetriever-style subquery bucket."""
    return {"subquery": subquery, "priority": 1, "results": sources, "similar_results": []}


# Two corroborating sources that agree on a figure -> confirmed.
_CONFIRMED_BUCKET = _bucket(
    "quantum market size",
    [
        {
            "title": "Quantum Computing Market Report",
            "url": "https://example.com/a",
            "text": "The quantum computing market is projected to reach 47 billion dollars.",
        },
        {
            "title": "Quantum Industry Outlook",
            "url": "https://example.com/b",
            "text": "Analysts size the quantum computing market at about 47 billion dollars.",
        },
    ],
)

# Two corroborating sources that cite divergent figures -> conflicting.
_CONFLICT_BUCKET = _bucket(
    "quantum revenue forecast",
    [
        {
            "title": "Quantum Revenue Report",
            "url": "https://example.com/a",
            "text": "The quantum computing revenue is projected to reach 47 billion dollars.",
        },
        {
            "title": "Rival Quantum Outlook",
            "url": "https://example.com/b",
            "text": "The quantum computing revenue is projected to reach 52 billion dollars.",
        },
    ],
)

@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# EvidenceStateTracker unit tests (new module)
# --------------------------------------------------------------------------- #


def test_need_with_no_sources_is_open():
    tracker = EvidenceStateTracker()
    tracker.register_needs_from_plan(json.dumps({"subqueries": [{"query": "q"}]}))

    state = tracker.classify()

    assert state.total == 1
    assert state.needs[0].status == OPEN
    assert state.closure_ratio == 0.0


def test_single_source_is_open_below_corroboration_threshold():
    tracker = EvidenceStateTracker()
    tracker.commit_evidence([_bucket("quantum market size", _CONFIRMED_BUCKET["results"][:1])])

    state = tracker.classify()

    # One source cannot corroborate, so the need stays open (thin evidence).
    assert state.needs[0].status == OPEN
    assert state.needs[0].source_count == 1


def test_corroborating_sources_with_shared_figure_are_confirmed():
    tracker = EvidenceStateTracker()
    tracker.commit_evidence([_CONFIRMED_BUCKET])

    state = tracker.classify()

    assert state.needs[0].status == CONFIRMED
    assert state.conflicting == []
    assert state.closure_ratio == 1.0


def test_corroborating_sources_with_divergent_figures_conflict():
    tracker = EvidenceStateTracker()
    tracker.commit_evidence([_CONFLICT_BUCKET])

    state = tracker.classify()

    assert state.needs[0].status == CONFLICTING
    assert state.conflicting == state.needs
    assert "divergent" in state.needs[0].conflicts[0]


def test_non_corroborating_sources_stay_open():
    tracker = EvidenceStateTracker()
    bucket = _bucket(
        "mixed topic",
        [
            {"title": "Quantum computing advances", "url": "https://x.io/1", "text": "quantum computing qubits"},
            {"title": "Deep sea fishing rules", "url": "https://x.io/2", "text": "deep sea fishing regulations"},
        ],
    )
    tracker.commit_evidence([bucket])

    state = tracker.classify()

    # Sources share no salient terms -> fragmented, not corroborated.
    assert state.needs[0].status == OPEN


def test_buckets_match_registered_needs_by_subquery():
    tracker = EvidenceStateTracker()
    plan = json.dumps({"subqueries": [{"query": "Quantum Market Size"}, {"query": "Qubit Fidelity"}]})
    tracker.register_needs_from_plan(plan)
    # Bucket subquery text differs only by case/whitespace from the plan query.
    tracker.commit_evidence([_bucket("quantum market size", _CONFIRMED_BUCKET["results"])])

    state = tracker.classify()

    by_query = {n.query: n for n in state.needs}
    assert "Quantum Market Size" in by_query
    assert "Qubit Fidelity" in by_query
    assert by_query["Quantum Market Size"].status == CONFIRMED
    assert by_query["Qubit Fidelity"].status == OPEN  # no evidence committed


def test_closure_ratio_mixed_across_needs():
    tracker = EvidenceStateTracker()
    tracker.commit_evidence([_CONFIRMED_BUCKET, _CONFLICT_BUCKET])
    # A third need with no evidence, registered from an unlabeled bucket.
    tracker.commit_evidence([{"subquery": "orphan topic", "results": [], "similar_results": []}])

    state = tracker.classify()

    # Two distinct subquery texts above + one orphan -> 3 needs, 1 confirmed.
    assert state.total == 3
    assert len(state.confirmed) == 1
    assert len(state.conflicting) == 1
    assert len(state.open_needs) == 1
    assert state.closure_ratio == pytest.approx(1 / 3)


def test_render_lists_section_and_open_needs():
    tracker = EvidenceStateTracker()
    tracker.register_needs_from_plan(json.dumps({"subqueries": [{"query": "Qubit Fidelity"}]}))
    state = tracker.classify()
    rendered = tracker.render(state)

    assert "## Evidence State" in rendered
    assert "arXiv:2607.11433" in rendered
    assert "Open evidence needs" in rendered
    assert "no sources gathered" in rendered


def test_render_empty_state_is_graceful():
    rendered = EvidenceStateTracker().render()

    assert "## Evidence State" in rendered
    assert "No evidence needs were recorded" in rendered


def test_reset_clears_previous_run():
    tracker = EvidenceStateTracker()
    tracker.register_needs_from_plan(json.dumps({"subqueries": [{"query": "first"}]}))
    tracker.reset()
    tracker.register_needs_from_plan(json.dumps({"subqueries": [{"query": "second"}]}))

    assert [n.query for n in tracker.needs] == ["second"]


def test_unparseable_plan_is_ignored():
    tracker = EvidenceStateTracker()
    tracker.register_needs_from_plan("not json")

    assert tracker.needs == []


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_evidence_state(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.evidence_state, EvidenceStateTracker)
    assert supervisor.evidence_state.needs == []


def test_supervisor_planning_registers_open_needs(patch_config_keys):
    supervisor = SupervisorAgent()
    plan = json.dumps(
        {"status": "success", "subqueries": [{"query": "market size"}, {"query": "key players"}]}
    )
    supervisor.planning_agent.execute = lambda query: plan

    supervisor.execute_tool("planning_agent", {"research_query": "quantum computing"})

    queries = [n.query for n in supervisor.evidence_state.needs]
    assert queries == ["market size", "key players"]
    assert all(n.status == OPEN for n in supervisor.evidence_state.needs)


def test_supervisor_retrieval_commits_evidence_and_classifies(patch_config_keys):
    """Drive planning then retrieval through execute_tool and assert the state closes."""
    supervisor = SupervisorAgent()

    plan = json.dumps({"subqueries": [{"query": "quantum market size"}]})
    supervisor.planning_agent.execute = lambda query: plan
    supervisor.web_search_retriever.retrieve = lambda query, subq: "findings"
    supervisor.web_search_retriever.last_search_results = [_CONFIRMED_BUCKET]

    supervisor.execute_tool("planning_agent", {"research_query": "quantum computing"})
    supervisor.execute_tool(
        "web_search_retriever",
        {"research_query": "quantum computing", "subqueries_json": plan},
    )

    state = supervisor.evidence_state.classify()
    need = state.needs[0]
    assert need.query == "quantum market size"
    assert need.source_count == 2
    assert need.status == CONFIRMED


def test_supervisor_summarize_evidence_appends_section(patch_config_keys):
    supervisor = SupervisorAgent()
    plan = json.dumps({"subqueries": [{"query": "qubit fidelity"}]})
    supervisor.planning_agent.execute = lambda query: plan
    # No evidence committed -> the need stays open and surfaces in the section.
    supervisor.execute_tool("planning_agent", {"research_query": "quantum computing"})

    out = supervisor._summarize_evidence("# Report\n\nFindings.")

    assert out.startswith("# Report")
    assert "## Evidence State" in out
    assert "qubit fidelity" in out


def test_supervisor_research_resets_evidence_state(patch_config_keys):
    supervisor = SupervisorAgent()
    supervisor.evidence_state.register_needs_from_plan(
        json.dumps({"subqueries": [{"query": "stale"}]})
    )
    assert supervisor.evidence_state.needs  # precondition: non-empty

    supervisor.evidence_state.reset()

    assert supervisor.evidence_state.needs == []
