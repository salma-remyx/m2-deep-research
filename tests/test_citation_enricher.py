"""Tests for the inline citation enricher and its supervisor wiring.

The enricher is adapted from LongCite's inference-time Citation Search
(arXiv:2409.02897v3). These tests cover both the enricher in isolation and its
integration into the existing :class:`~src.agents.supervisor.SupervisorAgent`
-- the call site -- which is what proves the wiring actually invokes the new
code.
"""

import re

import pytest

from src.agents.citation_enricher import CitationEnricher

_LABEL_RE = re.compile(r"\[([^\]]*)\]\(https://example\.com/long\)")
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# Flat source fixtures (mirror ExaTool.format_results output).
GROUNDING_SOURCES = [
    {
        "title": "Quantum Computing Market Report",
        "url": "https://example.com/quantum-report",
        "text": "The quantum computing market is projected to reach 47 billion "
        "dollars by 2030 according to industry analysts.",
        "highlights": ["market projected to reach 47 billion"],
    },
    {
        "title": "Qubit Fidelity Research",
        "url": "http://www.example.com/qubit-fidelity/",
        "text": "Error rates in superconducting qubits have dropped below one "
        "percent over the last year.",
        "highlights": ["superconducting qubits error rates"],
    },
]


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# CitationEnricher unit tests (new module)
# --------------------------------------------------------------------------- #


def test_adds_citation_to_uncited_grounded_sentence():
    enricher = CitationEnricher()
    report = (
        "The quantum computing market is projected to reach 47 billion dollars "
        "by 2030."
    )
    enriched, result = enricher.cite(report, GROUNDING_SOURCES)

    assert result.citations_added == 1
    assert "https://example.com/quantum-report" in enriched
    assert "[Quantum Computing Market Report]" in enriched


def test_citation_lands_before_terminal_punctuation():
    enricher = CitationEnricher()
    report = "The quantum computing market is projected to reach 47 billion by 2030."
    enriched, _ = enricher.cite(report, GROUNDING_SOURCES)

    assert enriched.rstrip().endswith(").")


def test_leaves_already_cited_sentence_alone():
    enricher = CitationEnricher()
    report = (
        "The quantum market is strong "
        "[the report](https://example.com/quantum-report)."
    )
    enriched, result = enricher.cite(report, GROUNDING_SOURCES)

    assert result.citations_added == 0
    # The single existing citation is not duplicated.
    assert enriched.count("https://example.com/quantum-report") == 1


def test_no_citation_when_no_source_support():
    enricher = CitationEnricher()
    # A claim whose key terms appear in no retrieved source.
    report = "The Mars colony population reached 9 million settlers by 2077."
    enriched, result = enricher.cite(report, GROUNDING_SOURCES)

    assert result.citations_added == 0
    assert "https://example.com/quantum-report" not in enriched
    assert enriched == report


def test_skips_structural_markdown_lines():
    enricher = CitationEnricher()
    report = (
        "# Heading\n"
        "| Aspect | Value |\n"
        "|--------|-------|\n"
        "| Growth | 47B   |\n"
        "```\n"
        "code block with 47 billion inside\n"
        "```\n"
        "---"
    )
    enriched, result = enricher.cite(report, GROUNDING_SOURCES)

    assert result.citations_added == 0
    assert enriched == report


def test_skips_line_already_referencing_the_source():
    # A sources-list line that already carries the URL should not be re-cited.
    enricher = CitationEnricher()
    report = "- https://example.com/quantum-report — quantum market figures"
    enriched, result = enricher.cite(report, GROUNDING_SOURCES)

    assert result.citations_added == 0
    assert "[Quantum Computing Market Report]" not in enriched


def test_empty_sources_returns_report_unchanged():
    enricher = CitationEnricher()
    report = "The quantum market is projected to reach 47 billion by 2030."
    enriched, result = enricher.cite(report, [])

    assert result.verifiable is False
    assert enriched == report


def test_handles_nested_retriever_buckets():
    """Production shape: WebSearchRetriever returns subquery buckets, not flat sources."""
    enricher = CitationEnricher()
    nested = [
        {
            "subquery": "market size",
            "priority": 1,
            "results": [GROUNDING_SOURCES[0]],
            "similar_results": [],
        }
    ]
    report = "The quantum computing market is projected to reach 47 billion by 2030."
    enriched, result = enricher.cite(report, nested)

    assert result.sources_available == 1
    assert "https://example.com/quantum-report" in enriched


def test_dedupes_sources_by_normalized_url():
    enricher = CitationEnricher()
    dupes = [
        GROUNDING_SOURCES[0],
        {**GROUNDING_SOURCES[0], "title": "Duplicate Title"},  # same URL
    ]
    report = "The quantum computing market is projected to reach 47 billion by 2030."
    _, result = enricher.cite(report, dupes)

    assert result.sources_available == 1


def test_long_title_label_is_truncated():
    enricher = CitationEnricher()
    long_title = "A Very Long Source Title " * 10  # well over max_label_chars
    sources = [
        {
            "title": long_title,
            "url": "https://example.com/long",
            "text": "quantum computing market projected 47 billion 2030",
            "highlights": [],
        }
    ]
    report = "The quantum computing market is projected to reach 47 billion by 2030."
    enriched, _ = enricher.cite(report, sources)

    # The label is truncated to max_label_chars and carries the ellipsis.
    match = _LABEL_RE.search(enriched)
    assert match is not None
    label = match.group(1)
    assert label.endswith("…")
    assert len(label) <= enricher.max_label_chars


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_citation_enricher(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.citation_enricher, CitationEnricher)


def test_supervisor_enrich_citations_adds_grounded_citation(patch_config_keys):
    supervisor = SupervisorAgent()
    # Simulate sources captured from a web_search_retriever tool call.
    supervisor._gathered_sources = GROUNDING_SOURCES

    report = (
        "# Quantum Outlook\n\n"
        "The quantum computing market is projected to reach 47 billion dollars "
        "by 2030.\n"
    )
    enriched = supervisor._enrich_citations(report)

    # The report structure is preserved and a grounded inline citation added.
    assert enriched.startswith("# Quantum Outlook")
    assert "https://example.com/quantum-report" in enriched
    assert "[Quantum Computing Market Report]" in enriched


def test_supervisor_enrich_citations_skips_cleanly_with_no_sources(patch_config_keys):
    supervisor = SupervisorAgent()
    # No sources captured -> nothing to cite, report returned intact.
    report = "# Report\n\nThe quantum market is projected to reach 47 billion by 2030."
    enriched = supervisor._enrich_citations(report)

    assert enriched == report
    assert "https://example.com" not in enriched
