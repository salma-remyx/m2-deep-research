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

from src.agents.middle_resurfacer import (
    MiddleResurfacer,
    ResurfacedContext,
    position_penalty,
)
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
# A second, equally query-relevant source (same lexical overlap + highlight
# boost as RELEVANT) used to test position-aware restatement selection.
RELEVANT_B = {
    "title": "Qubit Fidelity Records",
    "url": "https://x.io/qubit2",
    "text": "qubit fidelity records broken this year",
    "highlights": ["qubit fidelity records"],
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
# U-shaped position penalty (ported from RAL-Writer's position_func.py)
# --------------------------------------------------------------------------- #


def test_position_penalty_is_u_shaped():
    # Zero at the center, rising to b=0.3 at both edges (a=60, b=0.3).
    assert position_penalty(0.5) == 0.0
    assert position_penalty(0.0) == pytest.approx(0.3)
    assert position_penalty(1.0) == pytest.approx(0.3)
    # Nearly flat through the middle, rising sharply only near the edges.
    assert position_penalty(0.4) < 1e-6
    assert position_penalty(0.1) > position_penalty(0.4)
    # Symmetric around the center and clamped outside [0, 1].
    assert position_penalty(0.2) == pytest.approx(position_penalty(0.8))
    assert position_penalty(-1.0) == pytest.approx(0.3)


def test_restatement_prefers_middle_positioned_source_over_equal_edge_source():
    """The paper's core mechanism: relevance MINUS the U-shaped penalty.

    Two sources with identical lexical relevance; the one buried in the middle
    of the original sequence must be restated ahead of the one already at an
    attended edge. Under the prior relevance-only ranking the edge source would
    win the tie by stable order.
    """
    resurfacer = MiddleResurfacer(key_sources_count=1)
    # RELEVANT at the leading edge (position 0), RELEVANT_B dead center.
    results = [_bucket([RELEVANT, FILLER[0], RELEVANT_B, FILLER[1], FILLER[0]])]

    out = resurfacer.resurface("qubit fidelity", results)

    assert out.key_sources[0]["url"] == "https://x.io/qubit2"


def test_resurface_edge_reorders_similar_results_too():
    resurfacer = MiddleResurfacer()
    bucket = _bucket([FILLER[0]])
    bucket["similar_results"] = [FILLER[0], RELEVANT, FILLER[1]]

    out = resurfacer.resurface("qubit fidelity", [bucket])

    similar = out.reordered_search_results[0]["similar_results"]
    assert similar[0] is RELEVANT or similar[-1] is RELEVANT
    assert similar[1] is not RELEVANT


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


def test_synthesize_findings_includes_similar_results_in_context(monkeypatch, patch_config_keys):
    retriever = WebSearchRetriever()

    captured = {}

    def fake_post(self, url, json=None, headers=None):
        captured["payload"] = json
        return _FakeResponse({"choices": [{"message": {"content": "synthesized"}}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    bucket = _bucket([FILLER[0]])
    bucket["similar_results"] = [RELEVANT]
    out = retriever.synthesize_findings("qubit fidelity advances", [bucket])

    assert out == "synthesized"

    content = captured["payload"]["messages"][-1]["content"]
    # The similar_results are now part of the synthesis context (previously
    # retrieved but silently dropped), under a compact "Related sources" list.
    assert "Related sources" in content
    assert "Qubit Fidelity Breakthrough" in content


def test_synthesize_findings_restates_middle_source_over_equal_edge_source(
    monkeypatch, patch_config_keys
):
    """Integration proof of the position-aware restatement at the call site."""
    retriever = WebSearchRetriever()

    captured = {}

    def fake_post(self, url, json=None, headers=None):
        captured["payload"] = json
        return _FakeResponse({"choices": [{"message": {"content": "synthesized"}}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    # Equally relevant sources: RELEVANT at the leading edge, RELEVANT_B in the
    # middle. The preamble must restate the middle one first.
    search_results = [_bucket([RELEVANT, FILLER[0], RELEVANT_B, FILLER[1], FILLER[0]])]
    out = retriever.synthesize_findings("qubit fidelity", search_results)

    assert out == "synthesized"

    content = captured["payload"]["messages"][-1]["content"]
    preamble = content.split("## Subquery:", 1)[0]
    assert "Key sources" in preamble
    assert preamble.index("Qubit Fidelity Records") < preamble.index(
        "Qubit Fidelity Breakthrough"
    )
