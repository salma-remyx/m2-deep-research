"""Tests for the persistent source workspace and its supervisor wiring.

The workspace is adapted from *Fetch-then-Explore* (arXiv:2608.02097v1), which
decouples page selection from evidence extraction and keeps fetched pages in a
per-question workspace so evidence can be re-pulled on demand. These tests
cover the workspace in isolation and its integration into the existing
:class:`~src.agents.supervisor.SupervisorAgent` -- the call site -- which is
what proves the wiring actually invokes the new code.
"""

import json

import pytest

from src.agents.source_workspace import EvidenceHit, ExploreResult, SourceWorkspace
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring
from src.agents.web_search_retriever import WebSearchRetriever  # non-new module

# Nested subquery buckets, matching WebSearchRetriever.last_search_results shape.
SEARCH_RESULTS = [
    {
        "subquery": "quantum market size",
        "priority": 1,
        "results": [
            {
                "url": "https://example.com/quantum-report",
                "title": "Quantum Computing Market Report",
                "text": (
                    "The quantum computing market is projected to reach 47 "
                    "billion dollars by 2030 according to industry analysts. "
                    "Growth is driven by cryptography and materials science."
                ),
                "highlights": ["market projected to reach 47 billion by 2030"],
            },
            {
                "url": "https://example.com/qubit-fidelity",
                "title": "Qubit Fidelity Research",
                "text": (
                    "Error rates in superconducting qubits have dropped below "
                    "one percent over the last year."
                ),
                "highlights": ["superconducting qubits error rates"],
            },
        ],
        "similar_results": [
            {
                # Duplicate URL of the first result -> must be deduped, not re-added.
                # Its text is deliberately longer than the original excerpt so the
                # merge replaces it.
                "url": "https://example.com/quantum-report",
                "title": "Quantum Computing Market Report (mirror)",
                "text": (
                    "Quantum computing market expansion: a longer consolidated "
                    "passage that supersedes the original excerpt because it "
                    "carries more detail about the 2030 projection and the "
                    "cryptography and materials science growth drivers."
                ),
            }
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
# SourceWorkspace unit tests (new module)
# --------------------------------------------------------------------------- #


def test_record_flattens_buckets_and_dedupes_by_url():
    ws = SourceWorkspace()
    added = ws.record(SEARCH_RESULTS)

    # 2 distinct URLs -- the third entry duplicates the first.
    assert added == 2
    assert len(ws) == 2
    # The merge kept the longer text for the duplicated URL.
    quantum = next(s for s in ws.to_dict() if "quantum-report" in s["url"])
    assert "longer consolidated passage" in quantum["text"]


def test_record_accumulates_across_calls():
    ws = SourceWorkspace()
    ws.record(SEARCH_RESULTS)
    added = ws.record(
        [{"url": "https://example.com/new", "title": "New", "text": "fresh source"}]
    )
    assert added == 1
    assert len(ws) == 3


def test_explore_re_extracts_matching_evidence():
    ws = SourceWorkspace()
    ws.record(SEARCH_RESULTS)

    result = ws.explore("quantum market size projections")

    assert isinstance(result, ExploreResult)
    assert result.workspace_size == 2
    assert result.hits  # something matched
    top = result.hits[0]
    assert isinstance(top, EvidenceHit)
    assert top.url == "https://example.com/quantum-report"
    # Snippet carries the focused evidence, not the unrelated qubit page.
    assert "47" in top.snippet or "market" in top.snippet.lower()


def test_explore_no_match_is_graceful():
    ws = SourceWorkspace()
    ws.record(SEARCH_RESULTS)
    result = ws.explore(" totally unrelated giraffe diet ")

    assert result.hits == []
    assert "No cached sources" in result.render()


def test_explore_empty_workspace_is_graceful():
    ws = SourceWorkspace()
    result = ws.explore("anything")
    assert result.hits == []
    assert result.workspace_size == 0


def test_explore_revisits_accumulate_per_source():
    ws = SourceWorkspace()
    ws.record(SEARCH_RESULTS)
    ws.explore("quantum market")
    ws.explore("quantum market")
    assert ws.stats()["revisits"] >= 2
    assert ws.explore_count == 2


def test_dump_and_load_roundtrip(tmp_path):
    ws = SourceWorkspace()
    ws.record(SEARCH_RESULTS)
    path = tmp_path / "workspace.json"
    ws.dump(str(path))

    payload = json.loads(path.read_text())
    assert len(payload) == 2

    reloaded = SourceWorkspace()
    assert reloaded.load(str(path)) == 2
    assert len(reloaded) == 2


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_registers_explore_tool(patch_config_keys):
    supervisor = SupervisorAgent()
    names = [t["name"] for t in supervisor.tools]
    assert "explore_workspace" in names


def test_retriever_records_workspace_on_retrieve(patch_config_keys):
    """retrieve() records raw sources into the persistent workspace (selection)."""
    retriever = WebSearchRetriever()
    assert len(retriever.workspace) == 0

    # Stub the networked pieces so retrieve() records sources without any HTTP.
    retriever.search_with_subqueries = lambda subq: SEARCH_RESULTS
    retriever.synthesize_findings = lambda query, results: "findings"

    retriever.retrieve("quantum computing", '{"subqueries": [{"query": "q"}]}')

    assert len(retriever.workspace) == 2


def test_supervisor_explore_returns_evidence_with_no_exa_call(patch_config_keys):
    """The paper's defining move: re-extract from cached pages without re-fetching.

    Seed the supervisor's retriever workspace, tripwire Exa so any re-fetch
    fails the test, then dispatch explore_workspace and assert evidence comes
    back from the cache -- proving explore re-reads the workspace rather than
    issuing a new search.
    """
    supervisor = SupervisorAgent()
    retriever = supervisor.web_search_retriever
    retriever.workspace.record(SEARCH_RESULTS)

    def _no_exa(*args, **kwargs):
        raise AssertionError("explore_workspace must not call Exa")

    retriever.exa.search = _no_exa
    retriever.exa.find_similar = _no_exa

    result = supervisor.execute_tool(
        "explore_workspace", {"focus": "quantum market size"}
    )

    assert "workspace" in result.lower()
    assert "https://example.com/quantum-report" in result
    assert "No new web search" in result


def test_supervisor_explore_missing_focus_is_handled(patch_config_keys):
    supervisor = SupervisorAgent()
    result = supervisor.execute_tool("explore_workspace", {"focus": ""})
    assert "Error" in result
