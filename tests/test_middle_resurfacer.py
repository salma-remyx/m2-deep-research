"""Tests for the lost-in-the-middle resurfacer and its synthesis wiring.

The resurfacer is adapted from RAL-Writer / *Lost-in-the-Middle in Long-Text
Generation* (arXiv:2503.06868v1). These tests cover both the resurfacer in
isolation and its integration into the existing
:class:`~src.agents.web_search_retriever.WebSearchRetriever.synthesize_findings`
-- the call site -- which is what proves the wiring actually invokes the new
code.
"""

import httpx
import pytest

from src.agents.middle_resurfacer import MiddleResurfacer, ResurfacedContext
from src.agents.web_search_retriever import WebSearchRetriever  # non-new module -> proves wiring

# Result fixtures mirror ExaTool.format_results output. The query-relevant
# source is intentionally placed in the MIDDLE of each bucket so the test can
# assert it is resurfaced to an edge rather than left buried.
FILLER = [
    {"title": "Cooking with Pasta", "url": "https://x.io/cook",
     "text": "pasta sauce recipes for dinner", "highlights": []},
    {"title": "Gardening Basics", "url": "https://x.io/garden",
     "text": "tomatoes and basil in the garden", "highlights": []},
]
RELEVANT = {
    "title": "Qubit Fidelity Breakthrough",
    "url": "https://x.io/qubit",
    "text": "qubit fidelity improved dramatically this year",
    "highlights": ["qubit fidelity improved dramatically"],
}


def _bucket(results):
    return {"subquery": "qubit fidelity", "priority": 1,
            "results": results, "similar_results": []}


class _FakeResponse:
    """Stand-in for an httpx.Response so no real API call is made."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):  # noqa: D401 - mirrors httpx API
        pass

    def json(self):
        return self._payload


# --------------------------------------------------------------------------- #
# Resurfacer unit tests (new module)
# --------------------------------------------------------------------------- #


def test_score_ranks_relevant_result_above_filler():
    resurfacer = MiddleResurfacer()

    relevant = resurfacer.score("qubit fidelity advances", RELEVANT)
    filler = resurfacer.score("qubit fidelity advances", FILLER[0])

    assert relevant > filler
    assert filler == 0.0  # no query-term overlap, no highlights


def test_reorder_moves_relevant_result_out_of_the_middle():
    resurfacer = MiddleResurfacer()
    # Relevant source sandwiched between two filler results.
    middle = [FILLER[0], RELEVANT, FILLER[1]]
    scored = [(resurfacer.score("qubit fidelity", r), r) for r in middle]

    reordered = resurfacer.reorder_to_edges(scored)

    # The relevant result was in the middle (index 1); it must now lead or
    # close the block -- i.e. it is no longer between both filler results.
    assert reordered[0] is RELEVANT or reordered[-1] is RELEVANT
    assert reordered[1] is not RELEVANT


def test_reorder_preserves_all_results():
    resurfacer = MiddleResurfacer()
    items = [FILLER[0], RELEVANT, FILLER[1]]
    scored = [(resurfacer.score("qubit fidelity", r), r) for r in items]

    reordered = resurfacer.reorder_to_edges(scored)

    assert sorted(id(r) for r in reordered) == sorted(id(r) for r in items)
    assert len(reordered) == len(items)


def test_resurface_builds_key_sources_preamble_deduping_urls():
    resurfacer = MiddleResurfacer()
    # Same relevant URL appears in two buckets; preamble must list it once.
    results = [_bucket([FILLER[0], RELEVANT]), _bucket([RELEVANT, FILLER[1]])]

    out = resurfacer.resurface("qubit fidelity", results)

    assert isinstance(out, ResurfacedContext)
    assert "Key sources" in out.preamble
    assert "Qubit Fidelity Breakthrough" in out.preamble
    # De-duplicated: the relevant URL appears once among the key sources.
    assert sum(1 for s in out.key_sources if s["url"] == "https://x.io/qubit") == 1


def test_resurface_empty_input_is_safe():
    resurfacer = MiddleResurfacer()
    out = resurfacer.resurface("anything", [])

    assert out.preamble == ""
    assert out.reordered_search_results == []


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing WebSearchRetriever
# --------------------------------------------------------------------------- #


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let WebSearchRetriever() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


def test_synthesize_findings_resurfaces_middle_content(monkeypatch, patch_config_keys):
    retriever = WebSearchRetriever()

    captured = {}

    def fake_post(self, url, json=None, headers=None):
        captured["payload"] = json
        return _FakeResponse({"choices": [{"message": {"content": "synthesized"}}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    # Relevant source buried in the middle of the only subquery bucket.
    search_results = [_bucket([FILLER[0], RELEVANT, FILLER[1]])]
    out = retriever.synthesize_findings("qubit fidelity advances", search_results)

    # The retriever still returns the model's synthesized text.
    assert out == "synthesized"

    content = captured["payload"]["messages"][-1]["content"]

    # (1) Restate move: the relevant source is restated up front in a preamble.
    assert "Key sources" in content
    assert "Qubit Fidelity Breakthrough" in content
    preamble_end = content.index("Key sources")

    # (2) Edge-reorder move: within its subquery block the relevant result now
    # leads the block instead of being sandwiched between the two filler
    # results.
    block = content.split("## Subquery:", 1)[1]
    qubit = block.index("Qubit Fidelity Breakthrough")
    cook = block.index("Cooking with Pasta")
    garden = block.index("Gardening Basics")

    assert qubit < cook and qubit < garden  # no longer lost-in-the-middle
    assert preamble_end < content.index("## Subquery:")  # preamble precedes blocks
