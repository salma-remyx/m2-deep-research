"""Tests for the recursive deep-and-wide search delegator and its wiring.

The delegator is a Mode 2 adapted port of WebSwarm (arXiv:2607.08662v1).
These tests cover the delegator in isolation and its integration into the
existing :class:`~src.agents.web_search_retriever.WebSearchRetriever` -- the
call site -- which is what proves the wiring actually invokes the new code.
"""

import json

import pytest

from src.agents.recursive_search import (
    MODE_DEEP,
    MODE_ENTITY,
    MODE_WIDE,
    RecursiveSearchDelegator,
)
from src.agents.web_search_retriever import WebSearchRetriever  # non-new module
from src.agents.auditor import ReportAuditor
from src.tools.exa_tool import ExaTool
from src.utils.config import Config

# Expected keys on every evidence bucket (a superset of the
# search_with_subqueries shape, which the auditor and synthesizer consume).
BUCKET_KEYS = {
    "subquery",
    "priority",
    "results",
    "similar_results",
    "mode",
    "depth",
    "children",
}


def _canned_search(query, num_results=10, **_kwargs):
    """Deterministic Exa-style raw responses keyed off the query string.

    Bodies repeat each vendor as a standalone Capitalized token (sentence- or
    comma-separated) so the entity-gap heuristic sees frequency >= 2 per vendor,
    matching how real result bodies repeat proper nouns. Highlights are kept
    lowercase so they do not merge vendors into one multi-word entity.
    """
    q = (query or "").lower()
    if "openai" in q:
        return {"results": [
            {"title": "openai research notes", "url": "https://q.example/openai",
             "text": "OpenAI develops foundation models. OpenAI works alongside "
                     "IBM on alignment.",
             "highlights": ["applied ai research"]},
        ]}
    if "ibm" in q:
        return {"results": [
            {"title": "ibm quantum notes", "url": "https://q.example/ibm",
             "text": "IBM partners with OpenAI on hybrid systems. IBM and OpenAI "
                     "collaborate broadly.",
             "highlights": ["strategic partnership"]},
        ]}
    if "google" in q:
        return {"results": [
            {"title": "google quantum notes", "url": "https://q.example/google",
             "text": "Google builds superconducting processors. Google is a "
                     "market leader.",
             "highlights": ["market leader"]},
        ]}
    if "rigetti" in q:
        return {"results": [
            {"title": "rigetti computing notes", "url": "https://q.example/rigetti",
             "text": "Rigetti manufactures quantum chips. Rigetti is an emerging "
                     "vendor.",
             "highlights": ["emerging vendor"]},
        ]}
    # Default: the root / probe query, surfacing three named vendors.
    return {"results": [
        {"title": "quantum computing landscape", "url": "https://q.example/root",
         "text": "IBM leads in superconducting qubits. Google leads with its "
                 "roadmap. Rigetti competes globally. IBM, Google, and Rigetti "
                 "are the key vendors.",
         "highlights": ["key vendors in the market"]},
    ]}


def _canned_similar(url, num_results=5, **_kwargs):
    slug = (url or "").rstrip("/").split("/")[-1] or "x"
    # Lowercase content so similar results never introduce spurious entities.
    return {"results": [
        {"title": f"adjacent coverage {slug}", "url": f"https://q.example/similar/{slug}",
         "text": f"additional context for {slug}.", "highlights": []},
    ]}


@pytest.fixture
def fake_exa(monkeypatch):
    """A real ExaTool with offline search/find_similar, keeping format_results real."""
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")
    exa = ExaTool()
    exa.search = _canned_search
    exa.find_similar = _canned_similar
    return exa


# --------------------------------------------------------------------------- #
# Delegator unit tests (new module)
# --------------------------------------------------------------------------- #


def test_delegate_returns_bucket_shape_and_modes(fake_exa):
    delegator = RecursiveSearchDelegator(fake_exa, max_depth=0, node_budget=12)
    buckets = delegator.delegate(
        "quantum computing",
        [
            {"query": "quantum research papers", "type": "research paper", "priority": 1},
            {"query": "quantum news", "type": "news", "priority": 2},
        ],
    )

    modes = {b["subquery"]: b["mode"] for b in buckets}
    assert modes["quantum research papers"] == MODE_DEEP
    assert modes["quantum news"] == MODE_WIDE

    for bucket in buckets:
        assert BUCKET_KEYS.issubset(bucket.keys())
        assert bucket["depth"] == 0
        assert bucket["children"] == []  # max_depth=0 disables delegation


def test_recursive_delegation_spawns_child_nodes(fake_exa):
    delegator = RecursiveSearchDelegator(fake_exa, max_depth=2, node_budget=12)
    buckets = delegator.delegate(
        "quantum computing",
        [{"query": "quantum computing overview", "type": "auto", "priority": 1}],
    )

    # The structure probe grounds entity_collect nodes for the vendors the root
    # surfaced (IBM / Google / Rigetti), and the IBM node recurses one level
    # deeper to collect OpenAI, which the root text never mentioned.
    entity_objectives = {b["subquery"] for b in buckets if b["mode"] == MODE_ENTITY}
    assert {"IBM", "Google", "Rigetti"}.issubset(entity_objectives)
    assert any("OpenAI" in b["children"] for b in buckets)

    # Child evidence aggregates upward into its parent bucket's results.
    ibm_bucket = next(b for b in buckets if b["subquery"] == "IBM")
    urls = {r["url"] for r in ibm_bucket["results"]}
    assert "https://q.example/ibm" in urls
    assert "https://q.example/openai" in urls


def test_node_budget_caps_total_searches(fake_exa):
    # Recursion is allowed (max_depth=2) but only two Exa calls are budgeted,
    # so no node can complete a child delegation and far fewer than the full
    # wide + 3 entity nodes are produced.
    delegator = RecursiveSearchDelegator(fake_exa, max_depth=2, node_budget=2)
    buckets = delegator.delegate(
        "quantum computing",
        [{"query": "quantum computing overview", "type": "auto", "priority": 1}],
    )

    assert all(bucket["children"] == [] for bucket in buckets)
    assert len(buckets) < 4


def test_dedupe_collapses_repeated_urls():
    # Coarse same-URL dedupe (case-insensitive, trailing slash normalized).
    out = RecursiveSearchDelegator._dedupe([
        {"url": "https://x.com/a", "title": "first"},
        {"url": "HTTPS://x.com/a/", "title": "dup"},
        {"url": "https://x.com/b", "title": "second"},
    ])
    assert len(out) == 2
    assert {r["title"] for r in out} == {"first", "second"}


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing WebSearchRetriever
# --------------------------------------------------------------------------- #


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let WebSearchRetriever() construct without real API credentials."""
    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


def test_retriever_wires_recursive_delegation(patch_config_keys):
    retriever = WebSearchRetriever()
    # Stub Exa so the delegator runs offline; stub synthesis to skip the network.
    retriever.exa.search = _canned_search
    retriever.exa.find_similar = _canned_similar
    retriever.synthesize_findings = lambda *a, **k: "SYNTHESIZED FINDINGS"

    payload = json.dumps({"subqueries": [
        {"query": "quantum computing overview", "type": "auto", "priority": 1},
    ]})
    findings = retriever.retrieve("quantum computing", payload)

    assert findings == "SYNTHESIZED FINDINGS"
    buckets = retriever.last_search_results
    assert buckets and BUCKET_KEYS.issubset(buckets[0].keys())

    # retrieve() routed through the recursive delegator: probe-grounded
    # entity_collect nodes are present alongside the planned wide node.
    assert any(b["mode"] == MODE_WIDE for b in buckets)
    assert any(b["mode"] == MODE_ENTITY for b in buckets)

    # The bucket contract is preserved: the auditor grounds a citation drawn
    # from the recursively gathered evidence.
    root_url = next(b for b in buckets if b["mode"] == MODE_WIDE)["results"][0]["url"]
    auditor = ReportAuditor()
    result = auditor.audit(f"Backed by [root]({root_url}).", buckets)
    assert result.unsupported_citations == []
