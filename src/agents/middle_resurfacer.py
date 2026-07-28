"""Mitigate lost-in-the-middle in the long synthesis context.

Adapted from **RAL-Writer** / *Lost-in-the-Middle in Long-Text Generation:
Synthetic Dataset, Evaluation Framework, and Mitigation* (arXiv:2503.06868v1).
That work shows long-input / long-output generation loses information buried in
the *middle* of the prompt, and mitigates it with an inference-time prompting
scheme that retrieves and restates important-but-middle-positioned content so
it is actually attended to (and cited).

This is a **Mode 2 (adapted port)** of that mitigation for this pipeline's
:meth:`~src.agents.web_search_retriever.WebSearchRetriever.synthesize_findings`,
which stuffs dozens of Exa excerpts into a single Gemini context:

* RAL-Writer's learned / attention-based "important content" detector is
  replaced by a **parameter-free lexical-overlap score** between the research
  query and each result's title/highlights/text, with a boost for results the
  retriever already flagged with highlights. It needs no model access and is
  fully deterministic, so it runs on every synthesis and is unit-testable
  offline.
* RAL-Writer's learned restatement generator is replaced by a deterministic
  **"Key sources" preamble** that restates the globally most relevant sources
  at the very front of the context.

**Core mechanism preserved (the lost-in-the-middle fix).** Two moves, both
inference-time and training-free: (1) **edge-reorder** -- within each subquery
block, results are reordered so the most query-relevant sit at the *start and
end* of the block (the positions long-context models attend to) and the least
relevant are pushed to the *middle*; and (2) **restate** -- a ``Key sources``
preamble explicitly restates the top sources up front so important content is
never buried, regardless of its original position. Together they surface and
restate the content that would otherwise be lost-in-the-middle.

**Scope of the port.** Only the *inference-time mitigation* is ported. The
paper's LongInOutBench benchmark suite and its synthetic long-input/long-output
dataset are not reproduced -- evaluation belongs in a downstream PR. The
importance signal is a lexical-overlap proxy, not a learned estimator, so it
approximates rather than replicates the paper's importance ranking. The score
is only used to decide *ordering*; every retrieved source stays in context.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Common words ignored when scoring relevance, so superficial keyword overlap
# does not read as "important".
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
    predicted research report reports analysis study findings result results
    overview summary source sources
    """.split()
)


@dataclass
class ResurfacedContext:
    """Output of resurfacing important-but-middle content before synthesis."""

    reordered_search_results: List[Dict[str, Any]] = field(default_factory=list)
    preamble: str = ""
    key_sources: List[Dict[str, Any]] = field(default_factory=list)


class MiddleResurfacer:
    """Reorder and restate retrieved results to fight lost-in-the-middle.

    Parameter-free: relevance is a lexical-overlap score between the research
    query and each result's text, plus a boost for results carrying retriever
    highlights. The score only decides *ordering*; no source is dropped.
    """

    def __init__(self, key_sources_count: int = 5, highlight_boost: float = 2.0):
        self.key_sources_count = key_sources_count
        self.highlight_boost = highlight_boost

    def score(self, query: str, result: Dict[str, Any]) -> float:
        """Lexical-overlap relevance of ``result`` to ``query`` (>= 0)."""
        q_terms = self._terms(query)
        if not q_terms:
            return 0.0
        title = str(result.get("title") or "")
        text = str(result.get("text") or "")
        highlights = result.get("highlights") or []
        if isinstance(highlights, list):
            hl_text = " ".join(str(h) for h in highlights)
        else:
            hl_text = ""
        doc_terms = self._terms(" ".join([title, text, hl_text]))
        relevance = float(len(q_terms & doc_terms))
        # Exa only emits highlights for topically-relevant passages; treat their
        # presence as a strong relevance signal (restates the paper's
        # "importance" detection via a target-native retriever cue).
        if isinstance(highlights, list) and any(str(h).strip() for h in highlights):
            relevance += self.highlight_boost
        return relevance

    def reorder_to_edges(self, scored: Sequence[Tuple[float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Move highest-scored items to the start and end; lowest to the middle.

        Long-context models attend best to the *edges* of the input, so the most
        query-relevant sources are placed there and the least relevant pushed to
        the middle (the lost-in-the-middle position). Original position breaks
        ties, so reordering is stable and deterministic.
        """
        indexed = list(enumerate(scored))
        indexed.sort(key=lambda pair: (-pair[1][0], pair[0]))
        ordered = [item for _idx, (_score, item) in indexed]
        front: List[Dict[str, Any]] = []
        back: List[Dict[str, Any]] = []
        for i, item in enumerate(ordered):
            # Even ranks lead the block; odd ranks close it (reversed, so the
            # second-most-relevant result ends up last / at the trailing edge).
            if i % 2 == 0:
                front.append(item)
            else:
                back.append(item)
        back.reverse()
        return front + back

    def resurface(self, query: str, search_results: Iterable[Any]) -> ResurfacedContext:
        """Reorder subquery buckets to the edges and build a key-sources preamble.

        Args:
            query: The research query the results were retrieved for.
            search_results: Nested subquery buckets from ``WebSearchRetriever``
                (each carrying a ``results`` list).

        Returns:
            A :class:`ResurfacedContext` with edge-reordered buckets (mirroring
            the input structure) and a ``Key sources`` preamble restating the
            globally most relevant, de-duplicated sources.
        """
        buckets: List[Dict[str, Any]] = []
        global_scored: List[Tuple[float, Dict[str, Any]]] = []
        for bucket in search_results:
            if not isinstance(bucket, dict):
                continue
            results = bucket.get("results") or []
            scored = [(self.score(query, r), r) for r in results if isinstance(r, dict)]
            reordered = self.reorder_to_edges(scored)
            new_bucket = dict(bucket)
            new_bucket["results"] = reordered
            buckets.append(new_bucket)
            global_scored.extend(scored)

        key_sources = self._top_sources(global_scored)
        preamble = self._format_preamble(key_sources)
        return ResurfacedContext(
            reordered_search_results=buckets,
            preamble=preamble,
            key_sources=key_sources,
        )

    # -- internals ---------------------------------------------------------

    def _top_sources(self, scored: Sequence[Tuple[float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Globally most-relevant sources, de-duplicated by URL (stable order)."""
        ranked = sorted(scored, key=lambda pair: -pair[0])
        seen: Set[str] = set()
        top: List[Dict[str, Any]] = []
        for _score, item in ranked:
            url = str(item.get("url") or "").strip().lower()
            if url:
                if url in seen:
                    continue
                seen.add(url)
            top.append(item)
            if len(top) >= self.key_sources_count:
                break
        return top

    def _format_preamble(self, key_sources: List[Dict[str, Any]]) -> str:
        if not key_sources:
            return ""
        lines = ["## Key sources (restated up front to avoid lost-in-the-middle)", ""]
        for src in key_sources:
            title = str(src.get("title") or "Untitled").strip()
            url = str(src.get("url") or "").strip()
            head = f"- **{title}**"
            if url:
                head += f" — {url}"
            lines.append(head)
            highlights = src.get("highlights") or []
            if isinstance(highlights, list):
                for hl in highlights[:2]:
                    if str(hl).strip():
                        lines.append(f"  - {str(hl).strip()}")
        lines.append("")
        return "\n".join(lines)

    def _terms(self, text: str) -> Set[str]:
        return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS
