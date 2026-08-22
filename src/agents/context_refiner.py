"""Distill retrieved context before synthesis -- refine context, then generate.

Adapted from the **distill-based context refiner** in *Mitigating Context
Interference for Reliable and Efficient Search Agents* (CRRL,
arXiv:2608.10743v1). CRRL studies multi-turn search agents and shows that the
interference degrading generation comes overwhelmingly from the *latest*
retrieved documents, and that distilling that context immediately before
generation -- "refine context and then generate" -- restores both reliability
and efficiency.

This is a **Mode 2 (adapted port)** for this deep-research pipeline:

* CRRL's refiner is a small LLM *distilled* from a larger teacher and trained
  into the agent's RL loop. This repo has no training infrastructure, so the
  distilled model is replaced by a **parameter-free query-relevance
  distiller**: sentence-level lexical overlap with the research query and its
  subquery, plus near-duplicate suppression. This is the same substitution
  BrainPilot's LLM fabrication judge underwent in ``src/agents/auditor.py``,
  and it keeps the pass deterministic, API-free, and unit-testable offline.
* CRRL's third contribution -- folding context refinement into the RL training
  pipeline -- is deliberately **not** ported: there is no training loop here
  to fold it into.
* Turn mapping: each ``WebSearchRetriever.retrieve()`` call gathers one fresh
  turn of Exa results and synthesizes from it immediately, so that turn *is*
  the "latest retrieved documents" CRRL identifies as the interference
  source. The refiner distills it in place, between
  ``search_with_subqueries()`` and ``synthesize_findings()``.

The retriever keeps its raw results untouched on ``last_search_results`` for
the post-synthesis grounding auditor; only the copy handed to synthesis is
distilled, so reports are audited against the full evidence while being
written from the refined context.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple


# Sentence boundary: split after . ! ? followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Content tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Words ignored when scoring a sentence against the query, so boilerplate
# overlap ("the report also notes...") does not read as "relevant".
_STOPWORDS: frozenset = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over
    under about as is are was were be been being this that these those it its
    their our your they them we you he she not no nor so than too very can
    could should would may might must will shall do does did has have had
    more most less least many much few several also however which who whom
    whose what when where why how during while across among between within
    without via per using used use new latest recent current currently
    report reports source sources according
    """.split()
)


@dataclass
class RefinementReport:
    """What the latest refinement pass stripped from the retrieved context."""

    sources_in: int = 0
    sources_kept: int = 0
    duplicates_dropped: int = 0
    chars_in: int = 0
    chars_out: int = 0

    @property
    def compression(self) -> float:
        """Fraction of retrieved characters that survived distillation."""
        if self.chars_in == 0:
            return 1.0
        return self.chars_out / self.chars_in


class ContextRefiner:
    """Distill the latest retrieved turn before it reaches the synthesis prompt.

    Two things are removed from each retrieved source, per CRRL's finding that
    the latest turn carries most of the interference:

    * **Irrelevant information** -- sentences whose content terms do not
      overlap the research query or the subquery that fetched them are dropped.
    * **Redundant details** -- sources whose distilled text is a near-duplicate
      (Jaccard similarity over content terms) of one already kept in the same
      subquery bucket are dropped, keeping the higher-ranked Exa result.

    Titles, URLs and highlights are passed through untouched, so citations and
    the downstream grounding audit are unaffected.
    """

    def __init__(
        self,
        min_token_overlap: int = 2,
        max_sentences: int = 4,
        max_source_chars: int = 600,
        duplicate_jaccard: float = 0.8,
    ):
        self.min_token_overlap = min_token_overlap
        self.max_sentences = max_sentences
        self.max_source_chars = max_source_chars
        self.duplicate_jaccard = duplicate_jaccard
        # Outcome of the most recent refine() call.
        self.last_report = RefinementReport()

    def refine(
        self,
        research_query: str,
        search_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return a distilled copy of ``search_results`` for the synthesis prompt.

        Args:
            research_query: The original research query the results serve.
            search_results: Subquery buckets emitted by
                ``WebSearchRetriever.search_with_subqueries``.

        Returns:
            New buckets in the same shape, with per-source text distilled to
            its query-relevant sentences and near-duplicate sources removed.
            The input is never mutated.
        """
        report = RefinementReport()
        query_tokens = self._content_tokens(research_query)
        refined_buckets: List[Dict[str, Any]] = []

        for bucket in search_results:
            if not isinstance(bucket, dict):
                continue
            # The subquery is what actually fetched these results, so a
            # sentence is relevant if it matches either it or the top-level
            # research query.
            target = query_tokens | self._content_tokens(str(bucket.get("subquery", "")))

            refined_bucket = dict(bucket)
            for key in ("results", "similar_results"):
                items = bucket.get(key)
                if not isinstance(items, list):
                    continue
                kept = self._refine_items(items, target, report)
                refined_bucket[key] = kept
            refined_buckets.append(refined_bucket)

        self.last_report = report
        return refined_buckets

    # -- internals ---------------------------------------------------------

    def _refine_items(
        self,
        items: List[Any],
        target: Set[str],
        report: RefinementReport,
    ) -> List[Dict[str, Any]]:
        """Distill one bucket's source list, updating ``report`` in place."""
        kept: List[Dict[str, Any]] = []
        seen_token_sets: List[Set[str]] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            report.sources_in += 1
            original = str(item.get("text") or "")
            report.chars_in += len(original)

            distilled = self._distill_text(original, target)
            token_set = self._content_tokens(distilled)

            # Redundant detail: this source says what an already-kept one in
            # the same bucket says. Exa ranks by relevance, so first-seen wins.
            if any(self._jaccard(token_set, other) >= self.duplicate_jaccard
                   for other in seen_token_sets):
                report.duplicates_dropped += 1
                continue

            refined = dict(item)
            refined["text"] = distilled
            kept.append(refined)
            seen_token_sets.append(token_set)
            report.sources_kept += 1
            report.chars_out += len(distilled)

        return kept

    def _distill_text(self, text: str, target: Set[str]) -> str:
        """Reduce one source's text to its query-relevant sentences.

        Sentences are ranked by content-term overlap with ``target``; the top
        ``max_sentences`` are kept and re-ordered as they appeared so the
        excerpt still reads coherently. If nothing clears the overlap bar the
        opening sentence is kept, so an Exa-ranked source stays citable rather
        than collapsing to an empty block.
        """
        text = text.strip()
        if not text:
            return ""

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            sentences = [text]

        scored: List[Tuple[float, int, str]] = []
        for position, sentence in enumerate(sentences):
            overlap = len(target & self._content_tokens(sentence))
            if overlap >= self.min_token_overlap:
                scored.append((-float(overlap), position, sentence))

        if not scored:
            selected = [sentences[0]]
        else:
            scored.sort()
            selected = [s for _, _, s in scored[: self.max_sentences]]
            selected.sort(key=sentences.index)

        distilled = " ".join(selected)
        if len(distilled) > self.max_source_chars:
            clipped = distilled[: self.max_source_chars].rsplit(" ", 1)[0]
            distilled = clipped + "..."
        return distilled

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        """Content-term similarity of two texts; empty texts are never dupes."""
        if not a or not b:
            return 0.0
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    def _content_tokens(self, text: str) -> Set[str]:
        return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS
