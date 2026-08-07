"""Persistent source workspace for on-demand evidence re-extraction.

Adapted from **Fetch-then-Explore: Decoupling Selection from Extraction over a
Persistent Workspace for Search Agents** (arXiv:2608.02097v1). The paper's core
move is to separate *selecting* a page from *extracting* evidence from it, and
to keep what it selects: pages are recorded in a per-question workspace and
evidence is pulled from them on demand, repeatedly, as the agent's hypothesis
sharpens. Pages are not released when the agent moves on, so a page that turns
out to matter many turns later does not have to be fetched again -- the agent
returns to it and re-extracts. The paper traces its accuracy gains to exactly
this "return to a page after leaving it" behavior.

This is a **Mode 2 (adapted port)** of that mechanism for this deep-research
pipeline:

* The paper writes full fetched pages to the **filesystem**. This pipeline does
  not fetch full pages -- the ``WebSearchRetriever`` works from **Exa excerpts**
  (title / url / text / highlights), so the Exa excerpt is the unit of "a page"
  here. The workspace holds those excerpts out of the supervisor's context
  window on the long-lived retriever object (the same role the paper's
  filesystem store plays), with an optional JSON dump for inspection.
* The paper's **evidence extraction** is done by the agent backbone (an LLM) on
  demand. That is replaced here by a **parameter-free lexical extractor**: given
  the supervisor's current *focus*, the workspace re-ranks cached sources by
  term overlap and returns the best-matching snippet from each. It needs no
  extra API keys and is fully deterministic, so the re-extraction path runs
  offline and is unit-testable -- mirroring how this repo's grounding auditor
  substitutes an LLM judge with a lexical proxy (arXiv:2607.15079v1).

**What is deliberately NOT changed.** The existing ``retrieve`` ->
``synthesize_findings`` path still produces a fetch-time reading of every
source (the paper's "visit-and-read" baseline). This port adds the paper's
*persistent + on-demand* re-extraction surface *alongside* it rather than
ripping it out: the retriever now records raw sources into the workspace at
fetch time (selection, kept), and a new ``explore_workspace`` tool lets the
supervisor re-extract evidence from that cache for a sharpened focus later in
the run -- with **no new Exa call**, which is the reduction-in-fetches the
paper reports.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Sentence boundary split for snippet selection.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Words ignored when scoring a focus against source text.
_STOPWORDS: Set[str] = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over
    under about as is are was were be been being this that these those it its
    their our your his her they them we you he she him not no nor so than too
    very can could should would may might must will shall do does did has have
    had more most less least many much few several also however which who whom
    whose what when where why how during while across among between within
    without via per using used use new one two three first second next according
    based recent currently reportedly said says estimated projected expected
    """.split()
)


@dataclass
class EvidenceHit:
    """A single re-extracted evidence passage from a cached source."""

    url: str
    title: str
    score: float
    snippet: str
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class ExploreResult:
    """Outcome of one on-demand re-extraction pass over the workspace."""

    focus: str
    hits: List[EvidenceHit] = field(default_factory=list)
    workspace_size: int = 0

    @property
    def revisits(self) -> int:
        """Number of cached sources this pass re-extracted evidence from."""
        return len(self.hits)

    def render(self) -> str:
        """Render the re-extracted evidence as a tool-result string."""
        if not self.hits:
            return (
                f"No cached sources matched the focus '{self.focus}'. "
                f"The workspace holds {self.workspace_size} source(s); "
                "call web_search_retriever to fetch new pages first."
            )
        lines = [
            f"Re-extracted {len(self.hits)} evidence passage(s) from the "
            f"persistent workspace ({self.workspace_size} cached source(s)) "
            f"for focus: '{self.focus}'.",
            "No new web search was performed -- evidence was pulled from "
            "previously fetched pages.",
            "",
        ]
        for i, hit in enumerate(self.hits, 1):
            lines.append(f"[{i}] {hit.title}")
            lines.append(f"    {hit.url}")
            lines.append(f"    {hit.snippet}")
            lines.append("")
        return "\n".join(lines).rstrip()


class SourceWorkspace:
    """A persistent, out-of-context cache of fetched sources.

    Sources are recorded once at fetch time (selection) and kept for the
    lifetime of a research run, so the supervisor can return to a previously
    fetched page and re-extract evidence from it on demand (extraction)
    without fetching it again.
    """

    def __init__(
        self,
        max_hits: int = 8,
        snippet_chars: int = 320,
        min_term_overlap: int = 2,
    ) -> None:
        self.max_hits = max_hits
        self.snippet_chars = snippet_chars
        self.min_term_overlap = min_term_overlap
        # normalized url -> accumulated source dict
        self._sources: Dict[str, Dict[str, Any]] = {}
        # normalized url -> times re-extracted by explore()
        self._revisits: Dict[str, int] = {}
        self._explores = 0

    def __len__(self) -> int:
        return len(self._sources)

    @property
    def explore_count(self) -> int:
        """Number of on-demand explore passes run against this workspace."""
        return self._explores

    def stats(self) -> Dict[str, int]:
        """Counts useful for the reduction-in-fetches metric."""
        return {
            "sources": len(self),
            "explores": self._explores,
            "revisits": sum(self._revisits.values()),
        }

    def record(self, search_results: Iterable[Any]) -> int:
        """Record fetched sources into the workspace.

        Accepts the nested subquery buckets emitted by
        ``WebSearchRetriever.search_with_subqueries`` or a flat list of source
        dicts. Sources are deduped by normalized URL and accumulated -- a page
        fetched under one subquery is still present when a later subquery or a
        later turn needs it.

        Returns the number of newly added sources.
        """
        added = 0
        for src in self._iter_sources(search_results):
            url = (src.get("url") or "").strip()
            if not url:
                continue
            key = self._normalize_url(url)
            if key not in self._sources:
                stored = dict(src)
                stored["url"] = url
                self._sources[key] = stored
                self._revisits.setdefault(key, 0)
                added += 1
            else:
                self._merge(self._sources[key], src)
        return added

    def explore(self, focus: str) -> ExploreResult:
        """Re-extract evidence for ``focus`` from cached sources (no fetch).

        This is the paper's defining move: return to previously fetched pages
        and pull the evidence that matters *now*, as the hypothesis sharpens.
        """
        self._explores += 1
        focus_terms = self._tokenize(focus) - _STOPWORDS
        hits: List[EvidenceHit] = []
        for key, src in self._sources.items():
            score, snippet, matched = self._score_source(src, focus_terms)
            if matched:
                self._revisits[key] += 1
                hits.append(
                    EvidenceHit(
                        url=src.get("url", ""),
                        title=src.get("title", "No title"),
                        score=score,
                        snippet=snippet,
                        matched_terms=sorted(matched),
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return ExploreResult(
            focus=focus, hits=hits[: self.max_hits], workspace_size=len(self)
        )

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return the accumulated sources as a serializable list."""
        return list(self._sources.values())

    def dump(self, path: str) -> None:
        """Persist the workspace to ``path`` as JSON (the paper's filesystem store)."""
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    def load(self, path: str) -> int:
        """Load sources from a JSON dump written by :meth:`dump`."""
        data = json.loads(Path(path).read_text())
        return self.record(data)

    # -- internals ---------------------------------------------------------

    def _iter_sources(self, sources: Iterable[Any]) -> Iterator[Dict[str, Any]]:
        """Yield flat source dicts, flattening retriever subquery buckets."""
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

    def _merge(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        """Keep the richer text when the same URL is fetched again."""
        if len(incoming.get("text") or "") > len(existing.get("text") or ""):
            existing["text"] = incoming.get("text") or ""
        highlights = incoming.get("highlights")
        if isinstance(highlights, list) and highlights:
            merged: List[Any] = list(existing.get("highlights") or [])
            for h in highlights:
                if h not in merged:
                    merged.append(h)
            existing["highlights"] = merged

    def _score_source(
        self, src: Dict[str, Any], focus_terms: Set[str]
    ) -> Tuple[float, str, Set[str]]:
        """Score a cached source against the focus; return (score, snippet, matched)."""
        if not focus_terms:
            return 0.0, "", set()
        title = src.get("title") or ""
        text = src.get("text") or ""
        highlights = src.get("highlights")
        parts: List[str] = [title, text]
        if isinstance(highlights, list):
            parts.extend(str(h) for h in highlights)
        matched = focus_terms & self._tokenize("\n".join(parts))
        if len(matched) < self.min_term_overlap:
            return 0.0, "", set()
        score = len(matched) / float(len(focus_terms))
        snippet = self._best_snippet(text, highlights, focus_terms)
        return score, snippet, matched

    def _best_snippet(
        self, text: str, highlights: Any, focus_terms: Set[str]
    ) -> str:
        """Pick the passage in the source that best matches the focus."""
        # Prefer an Exa highlight that already carries a focus term.
        if isinstance(highlights, list):
            for h in highlights:
                hs = str(h)
                if focus_terms & self._tokenize(hs):
                    return self._trim(hs)
        if not text.strip():
            return ""
        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
        if not sentences:
            return self._trim(text)
        best_idx = 0
        best_score = -1
        for idx, s in enumerate(sentences):
            overlap = len(focus_terms & self._tokenize(s))
            if overlap > best_score:
                best_score = overlap
                best_idx = idx
        # Grow outward from the best sentence up to snippet_chars.
        lo = hi = best_idx
        snippet = sentences[best_idx]
        while len(snippet) < self.snippet_chars and (
            lo > 0 or hi < len(sentences) - 1
        ):
            if hi < len(sentences) - 1:
                hi += 1
                snippet = " ".join(sentences[lo : hi + 1])
            if len(snippet) < self.snippet_chars and lo > 0:
                lo -= 1
                snippet = " ".join(sentences[lo : hi + 1])
        return self._trim(snippet)

    def _trim(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self.snippet_chars:
            return text
        return text[: self.snippet_chars].rstrip() + "..."

    def _tokenize(self, text: str) -> Set[str]:
        return set(_TOKEN_RE.findall(text.lower()))

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for dedup (drop scheme/www/fragment/trailing slash)."""
        cleaned = url.strip().lower().split("#")[0]
        cleaned = re.sub(r"^https?://(www\.)?", "", cleaned)
        return cleaned.rstrip("/")
