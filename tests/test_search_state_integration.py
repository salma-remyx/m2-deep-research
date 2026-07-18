"""Integration tests for SOCM search-state wiring in the retrieval loop.

These tests import the EXISTING call-site module
(`src.agents.web_search_retriever`) and exercise
`WebSearchRetriever.search_with_subqueries` with a stubbed ExaTool so no
network is required. They assert the two behaviors the new
`SearchState` (SOCM) adds to the loop:

  * Evidence Graph  -> overlapping URLs across subqueries and
                       find_similar() are deduplicated.
  * Failure Memory   -> a search pattern that returns no evidence is
                       remembered, so an identical retry is skipped
                       without calling Exa again.
"""

from src.agents.search_state import SearchState
from src.agents.web_search_retriever import WebSearchRetriever


class _StubExa:
    """Drop-in stand-in for ExaTool that returns scripted payloads."""

    def __init__(self, searches, similar):
        # query string -> raw exa search payload
        self._searches = searches
        # raw exa findSimilar payload, returned for any url
        self._similar = similar
        self.search_calls = []

    def search(self, query, **kwargs):
        self.search_calls.append(query)
        return self._searches.get(query, {"results": []})

    def find_similar(self, url, **kwargs):
        return self._similar

    @staticmethod
    def format_results(results):
        """Mirror ExaTool.format_results for the scripted payloads."""
        if "error" in results or "results" not in results:
            return []
        out = []
        for result in results.get("results", []):
            out.append(
                {
                    "title": result.get("title", "t"),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0),
                    "highlights": [],
                    "text": "",
                    "summary": "",
                }
            )
        return out


def _make_retriever(searches, similar):
    """Build a WebSearchRetriever whose Exa client is the stub."""
    retriever = WebSearchRetriever()
    retriever.exa = _StubExa(searches, similar)
    return retriever


def _urls(entry):
    """Collect every URL a subquery result entry contributed."""
    urls = [r.get("url", "") for r in entry.get("results", [])]
    urls += [r.get("url", "") for r in entry.get("similar_results", [])]
    return [u for u in urls if u]


def test_evidence_graph_dedupes_overlapping_urls_across_subqueries():
    # Subquery "alpha" finds {A}; find_similar returns {B, A} (A overlaps).
    # Subquery "beta" finds {A, C} (A overlaps alpha) plus an intra-list dup.
    searches = {
        "alpha": {"results": [{"url": "http://x/A"}]},
        "beta": {
            "results": [
                {"url": "http://x/A"},
                {"url": "http://x/C"},
                {"url": "http://x/C"},
            ]
        },
    }
    similar = {"results": [{"url": "http://x/B"}, {"url": "http://x/A"}]}

    retriever = _make_retriever(searches, similar)
    state = SearchState()
    results = retriever.search_with_subqueries(
        [
            {"query": "alpha", "type": "auto", "priority": 1},
            {"query": "beta", "type": "auto", "priority": 1},
        ],
        state=state,
    )

    # No URL should appear more than once across the whole run.
    all_urls = [u for entry in results for u in _urls(entry)]
    assert len(all_urls) == len(set(all_urls)), all_urls

    # The overlapping A and the intra-list C duplicate were both removed.
    assert state.stats["deduped"] >= 2, state.stats

    # find_similar still ran (priority 1 <= 3) and its overlapping A was
    # dropped against the primary search, leaving B as fresh evidence.
    assert any("http://x/B" == u for u in all_urls), all_urls

    # Nothing failed: both searches returned evidence, so Failure Memory
    # stayed empty and nothing was skipped.
    assert state.stats["failed"] == 0, state.stats
    assert state.stats["skipped_failed"] == 0, state.stats


def test_failure_memory_skips_repeated_empty_search_pattern():
    # Every search returns empty -> every distinct pattern is a failure.
    retriever = _make_retriever(searches={}, similar={"results": []})
    state = SearchState()
    results = retriever.search_with_subqueries(
        [
            {"query": "rare topic one", "type": "auto", "priority": 1},
            {"query": "rare topic one", "type": "auto", "priority": 2},
            {"query": "rare topic two", "type": "auto", "priority": 1},
        ],
        state=state,
    )

    # The first occurrence of each pattern was actually searched; the
    # duplicate "rare topic one" was skipped without hitting Exa again.
    assert retriever.exa.search_calls == ["rare topic one", "rare topic two"], (
        retriever.exa.search_calls
    )

    # The skipped entry is flagged and carries no evidence.
    skipped = [entry for entry in results if entry.get("skipped_repeat_failure")]
    assert len(skipped) == 1, results
    assert skipped[0]["results"] == [] and skipped[0]["similar_results"] == []

    assert state.stats["failed"] == 2, state.stats
    assert state.stats["skipped_failed"] == 1, state.stats
