"""Search-Oriented Context Management (SOCM) state for the Exa retrieval loop.

Turns the implicit, in-flight progress of a single retrieval run into
explicit, shared state so that:

  * overlapping subqueries and find_similar() calls no longer pile up the
    same URLs (Evidence Graph -> cross-subquery evidence dedup), and
  * search patterns that yield no usable evidence leave a trace, so an
    identical retry is skipped instead of burning search budget again
    (Failure Memory).

Adapted from SearchOS-V1 (arXiv:2607.15257v1). The paper's core mechanism
is kept at full fidelity:

  - Evidence Graph -> a set of already-gathered URLs. This is a
    parameter-free, URL-overlap proxy for the paper's relational citation
    graph; it delivers the same "don't re-gather what you already have"
    signal without a learned relational store.
  - Failure Memory  -> a set of normalized query signatures whose search
    returned no evidence, so identical retries are skipped.

Intentionally out of scope for this single-agent loop (Mode 2 cuts of
auxiliaries the repo cannot host):

  - Pipeline-parallel sub-agent scheduler (no multi-agent runtime here).
  - Search Tool Middleware Harness that intercepts model/tool traffic
    (the hook is wired directly into the retrieval loop instead).
  - Hierarchical strategy/access skill system.
  - Relational schema completion with grounded citations.
"""

from typing import Any, Dict, List, Set
from urllib.parse import urlparse


def _normalize_url(url: str) -> str:
    """Normalize a URL for "same evidence" comparisons.

    Collapses scheme/host case and strips the trailing slash so the same
    page reached via different links counts once. The query string and
    fragment are dropped: Exa canonical URLs rarely carry meaning in the
    query, and dropping them lets a result found by primary search and
    again by find_similar() dedupe cleanly.
    """
    if not url:
        return ""
    parsed = urlparse(url.strip())
    if not parsed.netloc:
        return url.strip().lower()
    scheme = parsed.scheme.lower() or "http"
    return f"{scheme}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"


def _query_signature(subquery: Dict[str, Any]) -> str:
    """Stable signature for a subquery's search pattern.

    Two subqueries with the same signature target the same search; if one
    already failed, the other is a redundant retry. Constraints that do
    not change the result set (content type, time period, domain filters)
    are part of the signature.
    """
    query = " ".join((subquery.get("query") or "").lower().split())
    content_type = (subquery.get("type") or "auto").lower()
    time_period = (subquery.get("time_period") or "any").lower()
    include_domains = tuple(
        sorted(d.lower() for d in (subquery.get("include_domains") or []))
    )
    exclude_domains = tuple(
        sorted(d.lower() for d in (subquery.get("exclude_domains") or []))
    )
    return (
        f"{query}|type={content_type}|tp={time_period}"
        f"|in={include_domains}|ex={exclude_domains}"
    )


class SearchState:
    """Persistent, shared state for one retrieval run (SOCM, adapted)."""

    def __init__(self) -> None:
        # Evidence Graph: URLs already gathered this run.
        self.seen_evidence: Set[str] = set()
        # Failure Memory: signatures of searches that returned no evidence.
        self.failed_signatures: Set[str] = set()
        self.stats: Dict[str, int] = {
            "deduped": 0,
            "failed": 0,
            "skipped_failed": 0,
        }

    # -- Failure Memory -------------------------------------------------
    def known_failure(self, subquery: Dict[str, Any]) -> bool:
        """True if an identical search pattern already yielded no evidence."""
        return _query_signature(subquery) in self.failed_signatures

    def record_failure(self, subquery: Dict[str, Any]) -> None:
        """Remember that this search pattern returned no usable evidence."""
        sig = _query_signature(subquery)
        if sig not in self.failed_signatures:
            self.failed_signatures.add(sig)
            self.stats["failed"] += 1

    def note_skipped_failure(self) -> None:
        """Record that a subquery was skipped via Failure Memory."""
        self.stats["skipped_failed"] += 1

    # -- Evidence Graph -------------------------------------------------
    def dedupe(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop results whose URL is already in the evidence graph.

        Surviving URLs are registered as gathered evidence, so later
        subqueries (and their find_similar() calls) won't re-add them.
        Also collapses duplicate URLs within the same result list. The
        input records are never mutated.
        """
        kept: List[Dict[str, Any]] = []
        for result in results:
            key = _normalize_url(result.get("url", ""))
            if not key or key in self.seen_evidence:
                self.stats["deduped"] += 1
                continue
            self.seen_evidence.add(key)
            kept.append(result)
        return kept
