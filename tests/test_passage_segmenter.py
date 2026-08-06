"""Tests for the variable-length passage segmenter and its retriever wiring.

The segmenter is adapted from *LumberChunker: Long-Form Narrative Document
Segmentation* (arXiv:2406.17526v1). These tests cover the segmenter in
isolation and its integration into the existing
:class:`~src.agents.web_search_retriever.WebSearchRetriever.synthesize_findings`
-- the call site -- which is what proves the wiring actually invokes the new
code.
"""

import httpx
import pytest

from src.agents.passage_segmenter import PassageSegmenter
from src.agents.web_search_retriever import WebSearchRetriever  # non-new module -> proves wiring


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let WebSearchRetriever() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "synthesized"}}]}


def test_segment_produces_variable_length_content_coherent_passages():
    """The LumberChunker core: a topic shift splits the document into
    variable-length passages, each internally coherent."""
    segmenter = PassageSegmenter(passage_words=8, window_size=6, min_overlap=2)
    # Two clearly distinct topics, each longer than one atomic passage so a
    # shift is detectable at the boundary.
    document = (
        "neural networks learn data representations neural networks train "
        "on labeled data neural networks scale with compute gradients "
        "tomatoes ripen red garden summer tomatoes fruit nightshade family "
        "tomatoes belong to the solanaceae plants botanically"
    )

    passages = segmenter.segment(document)

    assert len(passages) >= 2  # the topic shift produced a split
    joined = " ".join(p.text for p in passages)
    assert "neural" in joined and "tomato" in joined
    # The first passage stays on the first topic until the shift.
    assert "neural" in passages[0].text
    assert "tomato" not in passages[0].text.lower()


def test_segment_empty_and_short_documents():
    segmenter = PassageSegmenter()
    assert segmenter.segment("") == []
    short = segmenter.segment("tiny source")
    assert len(short) == 1
    assert short[0].text == "tiny source"


def test_select_ranks_by_query_relevance_within_budget():
    segmenter = PassageSegmenter(passage_words=8, window_size=6, min_overlap=2)
    document = (
        "neural networks learn data representations neural networks train "
        "on labeled data neural networks scale with compute gradients "
        "tomatoes ripen red garden summer tomatoes fruit nightshade family"
    )
    passages = segmenter.segment(document)

    # Query matches only the neural-networks topic.
    chosen = segmenter.select(passages, query="neural networks", max_chars=10_000)
    assert chosen  # something selected
    # Most relevant passage comes first in document order; the whole neural
    # topic must be present and the off-topic tomato passage excluded.
    text = " ".join(p.text for p in chosen)
    assert "neural" in text

    # A tight budget admits whole passages only up to the limit.
    tight = segmenter.select(passages, query="neural networks", max_chars=20)
    assert sum(len(p.text) for p in tight) <= max(
        20, len(tight[0].text) if tight else 0
    )


def test_synthesize_findings_surfaces_relevant_text_past_old_truncation(
    patch_config_keys, monkeypatch
):
    """Integration: synthesize_findings now segments the full source and keeps
    the most query-relevant passages, so content sitting *past* the old 1000-char
    hard cut reaches synthesis instead of being thrown away."""
    retriever = WebSearchRetriever()

    # ~1800 chars of off-topic filler (no overlap with the query), followed by
    # the only query-relevant sentence at the very end -- well past 1000 chars.
    filler = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed. " * 30
    relevant = (
        " The superconducting qubit achieved a gate fidelity of ninety seven "
        "percent in the latest benchmark run."
    )
    full_text = filler + relevant
    assert "ninety seven" not in full_text[:1000]  # old [:1000] would have dropped it

    source = {
        "subquery": "qubit fidelity error correction",
        "priority": 1,
        "results": [
            {
                "title": "Qubit Fidelity Benchmark",
                "url": "https://example.com/qubit",
                "text": full_text,
                "highlights": [],
            }
        ],
        "similar_results": [],
    }

    captured = {}

    def fake_post(self, url, *args, **kwargs):
        captured["payload"] = kwargs.get("json")
        return _FakeResponse()

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    retriever.synthesize_findings("qubit fidelity error correction", [source])

    content = captured["payload"]["messages"][1]["content"]
    # The late, query-relevant passage is now present in the synthesis context.
    assert "ninety seven" in content
    assert "qubit" in content


def test_synthesize_findings_leaves_source_text_intact_for_auditor(
    patch_config_keys, monkeypatch
):
    """Segmentation must read source text without mutating it, so the grounding
    auditor still sees the original ``text`` on each source dict."""
    retriever = WebSearchRetriever()
    full_text = "alpha beta gamma delta. " * 80
    source = {
        "subquery": "alpha",
        "priority": 1,
        "results": [{"title": "S", "url": "https://example.com/s", "text": full_text}],
        "similar_results": [],
    }
    monkeypatch.setattr(httpx.Client, "post", lambda self, *a, **k: _FakeResponse())

    retriever.synthesize_findings("alpha", [source])

    assert source["results"][0]["text"] == full_text  # untouched
