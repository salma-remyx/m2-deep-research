"""Tests for the context refiner and its retriever wiring.

The refiner is adapted from CRRL's distill-based context refiner
(arXiv:2608.10743v1). These tests cover the refiner in isolation and its
integration into the existing
:class:`~src.agents.web_search_retriever.WebSearchRetriever` -- the call
site -- which is what proves the wiring actually invokes the new code.
"""

import pytest

from src.agents.context_refiner import ContextRefiner, RefinementReport
from src.agents.web_search_retriever import WebSearchRetriever  # non-new module

# Nested subquery buckets, matching WebSearchRetriever.search_with_subqueries
# output (ExaTool.format_results shape).
QUERY = "quantum computing market size projections"

RETRIEVED = [
    {
        "subquery": "quantum computing market size",
        "priority": 1,
        "results": [
            {
                "title": "Quantum Computing Market Report",
                "url": "https://example.com/quantum-report",
                "text": (
                    "The quantum computing market is projected to reach 47 billion "
                    "dollars by 2030 according to industry analysts. "
                    "Our newsletter also covers coffee recipes and office news. "
                    "Growth is driven by pharmaceutical and finance adoption."
                ),
            },
            {
                "title": "Quantum Computing Market Report (mirror)",
                "url": "https://example.com/quantum-mirror",
                "text": (
                    "The quantum computing market is projected to reach 47 billion "
                    "dollars by 2030 according to industry analysts."
                ),
            },
        ],
        "similar_results": [],
    }
]


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let WebSearchRetriever() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# ContextRefiner unit tests (new module)
# --------------------------------------------------------------------------- #


def test_drops_off_topic_sentences():
    refiner = ContextRefiner()
    refined = refiner.refine(QUERY, RETRIEVED)

    text = refined[0]["results"][0]["text"]
    assert "47 billion" in text
    assert "coffee recipes" not in text


def test_drops_near_duplicate_source_within_bucket():
    refiner = ContextRefiner()
    refined = refiner.refine(QUERY, RETRIEVED)

    urls = [r["url"] for r in refined[0]["results"]]
    assert "https://example.com/quantum-report" in urls
    assert "https://example.com/quantum-mirror" not in urls
    assert refiner.last_report.duplicates_dropped == 1


def test_preserves_title_url_and_highlights():
    refiner = ContextRefiner()
    refined = refiner.refine(QUERY, RETRIEVED)

    source = refined[0]["results"][0]
    assert source["title"] == "Quantum Computing Market Report"
    assert source["url"] == "https://example.com/quantum-report"
    assert source["text"] != RETRIEVED[0]["results"][0]["text"]


def test_does_not_mutate_input():
    refiner = ContextRefiner()
    original = RETRIEVED[0]["results"][0]["text"]

    refiner.refine(QUERY, RETRIEVED)

    assert RETRIEVED[0]["results"][0]["text"] == original


def test_report_tracks_compression():
    refiner = ContextRefiner()
    refiner.refine(QUERY, RETRIEVED)

    report = refiner.last_report
    assert isinstance(report, RefinementReport)
    assert report.sources_in == 2
    assert report.sources_kept == 1
    assert report.chars_out < report.chars_in
    assert 0.0 < report.compression < 1.0


def test_fully_irrelevant_source_keeps_opening_sentence():
    """A source with no query overlap stays citable instead of collapsing."""
    refiner = ContextRefiner()
    buckets = [
        {
            "subquery": "quantum computing market",
            "results": [
                {
                    "title": "Off-topic",
                    "url": "https://example.com/other",
                    "text": "An unrelated page about gardening tools and sheds.",
                }
            ],
        }
    ]

    refined = refiner.refine(QUERY, buckets)

    assert refined[0]["results"][0]["text"].startswith("An unrelated page")


def test_empty_text_stays_empty():
    refiner = ContextRefiner()
    buckets = [
        {"subquery": "q", "results": [{"title": "T", "url": "https://x.io/1", "text": ""}]}
    ]

    refined = refiner.refine(QUERY, buckets)

    assert refined[0]["results"][0]["text"] == ""


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing WebSearchRetriever
# --------------------------------------------------------------------------- #


def test_retriever_instantiates_refiner(patch_config_keys):
    retriever = WebSearchRetriever()

    assert isinstance(retriever.context_refiner, ContextRefiner)


def test_retrieve_distills_context_before_synthesis(patch_config_keys):
    """The wiring: search results are refined on the way into synthesis."""
    retriever = WebSearchRetriever()
    captured = {}

    retriever.search_with_subqueries = lambda subqueries: RETRIEVED
    retriever.synthesize_findings = lambda query, results: captured.update(
        results=results
    ) or "findings"

    out = retriever.retrieve(QUERY, '{"subqueries": [{"query": "q"}]}')

    assert out == "findings"
    # Synthesis received the distilled buckets, not the raw ones.
    synth = captured["results"]
    assert "coffee recipes" not in synth[0]["results"][0]["text"]
    assert [r["url"] for r in synth[0]["results"]] == ["https://example.com/quantum-report"]
    # The raw evidence is still exposed for the grounding auditor.
    assert retriever.last_search_results == RETRIEVED


def test_retrieve_raw_results_survive_for_auditor(patch_config_keys):
    retriever = WebSearchRetriever()
    retriever.search_with_subqueries = lambda subqueries: RETRIEVED
    retriever.synthesize_findings = lambda query, results: "findings"

    retriever.retrieve(QUERY, '{"subqueries": [{"query": "q"}]}')

    assert retriever.last_search_results[0]["results"][0]["text"].startswith(
        "The quantum computing market"
    )
