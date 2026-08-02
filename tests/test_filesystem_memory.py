"""Tests for the filesystem-based memory store and its supervisor wiring.

The memory model is a Mode 2 adapted port of *Filesystem-Based Memory for LLM
Agents* (arXiv:2607.26637v1). These tests cover both the store in isolation
(management + search roles, search economy) and its integration into the
existing :class:`~src.agents.supervisor.SupervisorAgent` -- the call site --
which is what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.filesystem_memory import FilesystemMemory, RecallResult
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# Flat source fixtures (mirror ExaTool.format_results output).
QUANTUM_SOURCE = {
    "title": "Quantum Computing Market Report",
    "url": "https://example.com/quantum-report",
    "text": "The quantum computing market is projected to reach 47 billion "
    "dollars by 2030 according to industry analysts.",
    "highlights": ["market projected to reach 47 billion"],
}
QUBIT_SOURCE = {
    "title": "Qubit Fidelity Research",
    "url": "http://www.example.com/qubit-fidelity/",
    "text": "Error rates in superconducting qubits have dropped below one "
    "percent over the last year.",
    "highlights": ["superconducting qubits error rates"],
}
CLIMATE_SOURCE = {
    "title": "Carbon Capture Outlook",
    "url": "https://example.com/carbon-capture",
    "text": "Direct air capture capacity is expanding to remove atmospheric "
    "carbon dioxide at gigaton scale.",
    "highlights": ["direct air capture capacity"],
}


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# Management role: integrate incoming sources into a hierarchy
# --------------------------------------------------------------------------- #


def test_integrate_flat_sources_writes_markdown_files(tmp_path):
    memory = FilesystemMemory(root=tmp_path / "memory")
    written = memory.integrate([QUANTUM_SOURCE, QUBIT_SOURCE], query="quantum computing")

    assert written == 2
    files = list((tmp_path / "memory").rglob("*.md"))
    assert len(files) == 2
    # Both files live under the query-derived topic directory.
    topics = {p.parent.name for p in files}
    assert topics == {"quantum-computing"}
    # Front matter carries the url so sources stay citeable.
    body = files[0].read_text()
    assert "url:" in body and "category:" in body


def test_integrate_nested_subquery_buckets_bins_by_subquery(tmp_path):
    """Production shape: WebSearchRetriever returns per-subquery buckets."""
    memory = FilesystemMemory(root=tmp_path / "memory")
    nested = [
        {
            "subquery": "quantum market size",
            "results": [QUANTUM_SOURCE],
            "similar_results": [],
        },
        {
            "subquery": "qubit error correction",
            "results": [QUBIT_SOURCE],
            "similar_results": [],
        },
    ]
    memory.integrate(nested, query="quantum computing")

    files = list((tmp_path / "memory").rglob("*.md"))
    topics = {p.parent.name for p in files}
    assert "quantum-market-size" in topics
    assert "qubit-error-correction" in topics


def test_integrate_skips_duplicate_urls(tmp_path):
    memory = FilesystemMemory(root=tmp_path / "memory")
    memory.integrate([QUANTUM_SOURCE], query="quantum")
    # Re-running the same query must not duplicate the source.
    second = memory.integrate([QUANTUM_SOURCE], query="quantum")

    assert second == 0
    stats = memory.stats()
    assert stats.files == 1
    assert stats.duplicates_skipped == 1


# --------------------------------------------------------------------------- #
# Search role + the paper's headline result: search economy
# --------------------------------------------------------------------------- #


def test_recall_returns_cited_sources_for_relevant_query(tmp_path):
    memory = FilesystemMemory(root=tmp_path / "memory")
    memory.integrate([QUANTUM_SOURCE, CLIMATE_SOURCE], query="mixed")

    result = memory.recall("quantum market dollars", limit=2)

    assert isinstance(result, RecallResult)
    assert result.entries
    # The quantum source outranks the unrelated climate source.
    assert "quantum" in result.entries[0].url.lower()
    cited = result.as_cited_sources()
    assert cited[0]["url"] == result.entries[0].url


def test_recall_on_empty_store_is_safe(tmp_path):
    memory = FilesystemMemory(root=tmp_path / "empty")
    result = memory.recall("anything")

    assert result.entries == []
    assert result.search_economy == 0.0


def test_organized_store_has_lower_retrieval_cost_than_flat(tmp_path):
    """The paper's core result: organized stores cut retrieval cost as the store grows."""
    memory = FilesystemMemory(root=tmp_path / "memory")
    # Two distinct topics so the query matches only a narrow slice.
    memory.integrate(
        [
            {"subquery": "quantum market", "results": [QUANTUM_SOURCE], "similar_results": []},
            {"subquery": "carbon capture", "results": [CLIMATE_SOURCE], "similar_results": []},
        ],
        query="research",
    )

    result = memory.recall("quantum market dollars", limit=1)

    # The organized search read strictly less than a flat dump would have...
    assert result.bytes_scanned < result.flat_bytes_scanned
    assert result.files_opened < result.flat_files_opened
    # ...which is exactly the "search economy" the paper reports.
    assert result.search_economy > 0.0


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_memory(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.memory, FilesystemMemory)


def test_execute_tool_organizes_sources_into_memory(patch_config_keys, tmp_path, monkeypatch):
    supervisor = SupervisorAgent()
    # Point the store at a temp dir so the test never touches the repo.
    supervisor.memory = FilesystemMemory(root=tmp_path / "memory")
    captured = [QUANTUM_SOURCE]
    supervisor.web_search_retriever.retrieve = lambda query, subq: "findings"
    supervisor.web_search_retriever.last_search_results = captured

    out = supervisor.execute_tool(
        "web_search_retriever",
        {"research_query": "quantum computing", "subqueries_json": '{"subqueries": []}'},
    )

    assert out == "findings"
    # The management role fired: sources landed in the filesystem tree.
    assert supervisor.memory.stats().files == 1


def test_supervisor_recall_returns_organized_sources(patch_config_keys, tmp_path):
    supervisor = SupervisorAgent()
    supervisor.memory = FilesystemMemory(root=tmp_path / "memory")
    supervisor.memory.integrate([QUANTUM_SOURCE, CLIMATE_SOURCE], query="mixed")

    recalled = supervisor.recall("quantum market dollars", limit=2)

    assert recalled
    assert "quantum" in recalled[0]["url"].lower()
