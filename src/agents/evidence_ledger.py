"""Structured Evidence Ledger for provenance-linked report grounding.

Adapted from **LedgerMind: Provenance-Constrained Multimodal Agentic
Reasoning with a Structured Evidence Ledger** (arXiv:2607.28374v1).

LedgerMind's core mechanism -- the part this module ports -- is to treat a
multi-step agent trajectory's tool outputs as a **Structured Evidence Ledger**:
each retrieved source is normalized into a ledger entry with a stable id, the
ledger is the trajectory's grounding state, and every downstream reasoning
claim is *linked to the specific ledger entry(ies)* that back it rather than
scored against the corpus as a flat yes/no. Claims that cite no active entry
are flagged as unsupported.

This is a **Mode 2 (adapted port)** for this text deep-research pipeline:

* LedgerMind's learned LLM grounding judge and its entity/numeric NER
  extraction are replaced by a **parameter-free lexical-overlap proxy**
  (key-term overlap against an entry's text). This matches the convention
  already used by this repo's :class:`~src.agents.auditor.ReportAuditor` --
  fully deterministic, needs no extra API keys, and runs on every report.
* LedgerMind's *Event-Triggered Verification-and-Repair* engine and its formal
  provenance non-amplification guarantee are **not** ported: the audit here is
  read-only -- it surfaces ungrounded claims and never rewrites the report --
  so there is no repair-time amplification surface to guard.
* The multimodal VQA framing is incidental (the paper's venue is VQA); the
  ledger / provenance scheme is modality-agnostic, and the Exa-retrieved web
  sources gathered by ``WebSearchRetriever`` are the natural evidence corpus.

What is preserved is the paper's core contribution: a Structured Evidence
Ledger that is the grounding state, with each audited claim carrying
provenance back to the specific retrieved source(s) that support it.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Common words ignored when matching a claim against a source, so superficial
# keyword overlap does not read as "grounded". Mirrors the auditor's stoplist.
_STOPWORDS: Set[str] = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over
    under about as is are was were be been being this that these those it its
    their our your his her their they them we you he she him her not no nor so
    than too very can could should would may might must will shall do does did
    has have had more most less least many much few several also however which
    who whom whose what when where why how during while across among between
    within without via per using used use new one two three first second next
    according based recent currently reportedly said says according estimated
    projected expected predicted
    """.split()
)


def tokenize(text: str) -> Set[str]:
    """Lowercase alphanumeric tokens of >= 3 chars."""
    return set(_TOKEN_RE.findall(text.lower()))


def normalize_url(url: str) -> str:
    """Normalize a URL for set comparison (drop scheme/www/fragment)."""
    cleaned = url.strip().lower().split("#")[0]
    cleaned = re.sub(r"^https?://(www\.)?", "", cleaned)
    return cleaned.rstrip("/")


def iter_flat_sources(sources: Iterable[Any]) -> Iterator[Dict[str, Any]]:
    """Yield flat source dicts, flattening ``WebSearchRetriever`` subquery buckets.

    Accepts either a flat list of source dicts (``url``/``title``/``text``/
    ``highlights``) or the nested ``results``/``similar_results`` buckets the
    retriever emits.
    """
    for src in sources:
        if not isinstance(src, dict):
            continue
        if "results" in src or "similar_results" in src:
            for bucket_key in ("results", "similar_results"):
                for item in src.get(bucket_key) or []:
                    if isinstance(item, dict):
                        yield item
        else:
            yield src


@dataclass(frozen=True)
class LedgerEntry:
    """One retrieved source normalized into a ledger entry.

    ``entry_id`` (e.g. ``"E1"``) is the stable handle a claim cites as its
    provenance -- LedgerMind's "active ledger entry". ``tokens`` is the entry's
    content key-terms (stopwords removed), the set a claim's terms are matched
    against.
    """

    entry_id: str
    url: str
    title: str
    text: str
    tokens: frozenset[str] = field(default_factory=frozenset)


class EvidenceLedger:
    """Structured Evidence Ledger built from retrieved sources.

    Each source becomes a :class:`LedgerEntry` with a stable id. The ledger is
    the grounding state: a citation URL resolves to an entry via
    :meth:`find_url`, and a claim resolves to its backing entries via
    :meth:`backing_entries`. A claim with no backing entry is unsupported.
    """

    def __init__(
        self,
        sources: Iterable[Any],
        min_token_overlap: int = 3,
    ) -> None:
        self.min_token_overlap = min_token_overlap
        entries: List[LedgerEntry] = []
        url_index: Dict[str, LedgerEntry] = {}
        for idx, src in enumerate(iter_flat_sources(sources), start=1):
            entry_id = f"E{idx}"
            raw_url = src.get("url") or ""
            norm_url = normalize_url(raw_url) if raw_url else ""
            parts = [src.get("title") or "", src.get("text") or ""]
            highlights = src.get("highlights")
            if isinstance(highlights, list):
                parts.append(" ".join(str(h) for h in highlights))
            text = " ".join(parts)
            entry = LedgerEntry(
                entry_id=entry_id,
                url=norm_url,
                title=str(src.get("title") or ""),
                text=text,
                tokens=frozenset(tokenize(text) - _STOPWORDS),
            )
            entries.append(entry)
            # First entry wins for a duplicate normalized URL, matching the
            # auditor's set-membership behavior.
            if norm_url:
                url_index.setdefault(norm_url, entry)
        self._entries: List[LedgerEntry] = entries
        self._url_index: Dict[str, LedgerEntry] = url_index

    @property
    def size(self) -> int:
        """Number of ledger entries (gathered sources)."""
        return len(self._entries)

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def find_url(self, url: str) -> Optional[LedgerEntry]:
        """Return the active ledger entry for a citation URL, or ``None``.

        This is the provenance lookup for a citation: which gathered source
        (if any) does this cited URL trace back to?
        """
        if not url:
            return None
        return self._url_index.get(normalize_url(url))

    def backing_entries(self, claim: str) -> List[LedgerEntry]:
        """The specific ledger entries whose key terms back ``claim``.

        Mirrors LedgerMind's grounding check at the term level: a claim is
        grounded iff it cites at least one active ledger entry. Here "term
        level" is a lexical-overlap proxy for the paper's learned judge -- a
        claim backs an entry when their shared content terms meet
        ``min_token_overlap``. The non-empty / empty split is exactly the
        grounded / unsupported decision, now with the backing entries attached
        as provenance.
        """
        claim_tokens = tokenize(claim) - _STOPWORDS
        if not claim_tokens:
            # Degenerate claim with no content tokens: trivially grounded by
            # every entry (mirrors the auditor's empty-overlap -> grounded).
            return list(self._entries)
        return [
            entry
            for entry in self._entries
            if len(claim_tokens & entry.tokens) >= self.min_token_overlap
        ]

    def active_entry_ids(self, cited_urls: Iterable[str]) -> Set[str]:
        """Ids of ledger entries actually referenced by the report's citations.

        The "active" subset of the ledger -- the gathered sources the report put
        to use -- as distinct from the full gathered set.
        """
        active: Set[str] = set()
        for url in cited_urls:
            entry = self.find_url(url)
            if entry is not None:
                active.add(entry.entry_id)
        return active
