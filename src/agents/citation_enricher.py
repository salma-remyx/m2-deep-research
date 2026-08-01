"""Post-synthesis inline citation enrichment for research reports.

Adapted from **LongCite** (Zhang et al., arXiv:2409.02897v3), which lets
long-context LLMs emit fine-grained **sentence-level** citations by running an
inference-time **Citation Search**: after each statement, the best supporting
context in the source documents is retrieved and attached as a citation. The
supervisor's system prompt here already *demands* inline ``[label](url)``
citations, and the :class:`~src.agents.auditor.ReportAuditor` exists only
because they are unreliable -- yet nothing in the pipeline actually *generates*
them. This module fills that one gap.

This is a **Mode 2 (adapted port)** of LongCite's inference-time citation
search for this deep-research pipeline:

* LongCite's **retriever** (BM25 / dense context-span retrieval over the
  source documents) is replaced by a **parameter-free lexical-overlap proxy**
  over the sources the ``WebSearchRetriever`` already gathered -- the same
  grounding signal the auditor trusts, so citation search needs no extra model,
  index, or API key and runs deterministically on every report.
* LongCite's **sentence-index citation format** ``[i]`` (a pointer into a
  numbered list of context spans) is mapped to this repo's native
  ``[label](url)`` source-text format -- the one format adaptation, exactly as
  the pipeline's reports already use it.

**Intentionally out of scope (cut for this Mode 2 port):**

* LongCite's **training** -- Citation-Friendly Fine-Tuning (CFT) and the
  LongCite-RL reinforcement stage -- which shapes the model to emit citations
  natively. We do not retrain the supervisor; we run citation search at
  inference over whatever report it produced, so the contribution lands without
  a training pipeline this repo cannot host.
* LongCite's **passage-span precision** -- it cites the exact supporting
  passage. The lexical proxy cites the best-matching **source document** (by
  URL), not a verbatim span. This is the auxiliary substitution; the core
  mechanism -- a per-sentence citation search that grounds each statement in
  retrieved evidence -- is preserved.
* LongCite's **LongBench / LongBench-Cite** evaluation suite. Evaluation
  belongs in a downstream PR; here we return a small result summary instead.

The core mechanism is preserved: an independent post-synthesis pass that, for
each report sentence, retrieves the most-supporting gathered source and -- when
one clears a grounding bar -- inserts a fine-grained inline citation the report
did not have. It complements the grounding auditor, which then *verifies* the
(now richer) citation set.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple


# Any URL -- bare or inside a markdown link -- so we can tell when a line
# already references a source and should not be re-cited.
_ANY_URL_RE = re.compile(r"https?://[^\s)\]]+")

# Inline markdown citation: [label](url). A sentence carrying one is already cited.
_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation / short noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Sentence boundary: terminal punctuation + whitespace, only when the next
# non-space char is uppercase or a digit. The lookahead avoids splitting on
# abbreviations such as "U.S.", "Dr.", "e.g.".
_SENTENCE_SPLIT_RE = re.compile(r"([.!?]+\s+)(?=[A-Z0-9])")

# Trailing terminal punctuation to insert a citation *before*.
_TERMINAL_RE = re.compile(r"[.!?]+$")

# A leading markdown list / blockquote marker to peel (and restore) per line.
_PREFIX_RE = re.compile(r"^(\s*(?:>\s+|[-*+]\s+|\d+\.\s+))(.*)$")

# Common words ignored when matching a sentence against a source, so that
# superficial keyword overlap does not trigger a spurious citation.
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
    predicted
    """.split()
)


@dataclass
class CitationResult:
    """Outcome of an inline citation enrichment pass over a single report."""

    citations_added: int = 0
    sentences_cited: int = 0
    sources_available: int = 0
    verifiable: bool = True

    @property
    def enriched(self) -> bool:
        """True when at least one grounded inline citation was inserted."""
        return self.verifiable and self.citations_added > 0


class CitationEnricher:
    """Insert fine-grained, sentence-level citations into a synthesized report.

    The enricher is deliberately parameter-free: given a report (markdown that
    may contain inline ``[text](url)`` citations) and the sources gathered by
    the retriever, it walks the report sentence by sentence and, for any
    *uncited* statement that shares enough key terms with a retrieved source,
    appends an inline citation to that source. Sentences already carrying a
    citation, structural markdown, and lines that already reference the
    candidate source are left untouched, so the report's existing citations and
    formatting are never mangled.
    """

    def __init__(
        self,
        min_token_overlap: int = 3,
        min_sentence_chars: int = 24,
        max_label_chars: int = 60,
    ):
        self.min_token_overlap = min_token_overlap
        self.min_sentence_chars = min_sentence_chars
        self.max_label_chars = max_label_chars

    def cite(self, report: str, sources: Iterable[Any]) -> Tuple[str, CitationResult]:
        """Enrich ``report`` with sentence-level citations from ``sources``.

        Args:
            report: Synthesized report text (may contain markdown citations).
            sources: Retrieved sources. May be a flat list of source dicts
                (``url`` / ``title`` / ``text`` / ``highlights``) or the nested
                subquery buckets emitted by ``WebSearchRetriever``.

        Returns:
            ``(enriched_report, result)`` -- the report with inline citations
            inserted where support was found, and a summary of what was added.
            When no sources were gathered the report is returned unchanged.
        """
        src_list = list(self._iter_sources(sources))
        corpus = self._build_corpus(src_list)
        if not corpus:
            return report, CitationResult(
                verifiable=bool(src_list), sources_available=0
            )

        lines = report.split("\n")
        new_lines: List[str] = []
        in_fence = False
        citations_added = 0
        sentences_cited = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                new_lines.append(line)
                continue
            if in_fence or not stripped or self._is_structural(stripped):
                new_lines.append(line)
                continue

            prefix, body = self._peel_prefix(line)
            new_body, added, matched = self._cite_prose(body, corpus)
            citations_added += added
            sentences_cited += matched
            new_lines.append(prefix + new_body)

        result = CitationResult(
            citations_added=citations_added,
            sentences_cited=sentences_cited,
            sources_available=len(corpus),
            verifiable=True,
        )
        return "\n".join(new_lines), result

    # -- internals ---------------------------------------------------------

    def _cite_prose(
        self, text: str, corpus: List[Dict[str, Any]]
    ) -> Tuple[str, int, int]:
        """Insert citations into ``text`` (a single prose line)."""
        if not text.strip():
            return text, 0, 0

        already_referenced = self._urls_present(text)
        parts = _SENTENCE_SPLIT_RE.split(text)
        added = 0
        matched = 0

        # Bodies sit at even indices; the captured separators at odd indices.
        for i in range(0, len(parts), 2):
            body = parts[i]
            if len(body.strip()) < self.min_sentence_chars:
                continue
            if _LINK_RE.search(body):  # already carries a citation
                continue
            best = self._best_source(body, corpus)
            if best is None:
                continue
            if self._normalize_url(best["url"]) in already_referenced:
                continue
            parts[i] = self._insert_citation(body, best)
            added += 1
            matched += 1

        return "".join(parts), added, matched

    def _best_source(
        self, sentence: str, corpus: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Return the source with the greatest key-term overlap, or None."""
        sent_tokens = self._tokenize(sentence) - _STOPWORDS
        if not sent_tokens:
            return None
        best: Optional[Dict[str, Any]] = None
        best_overlap = 0
        for src in corpus:
            overlap = len(sent_tokens & src["tokens"])
            if overlap > best_overlap:
                best_overlap = overlap
                best = src
        if best_overlap >= self.min_token_overlap:
            return best
        return None

    def _insert_citation(
        self, body: str, src: Dict[str, Any]
    ) -> str:
        """Append a citation to ``body`` immediately before its terminal punctuation."""
        citation = f"[{src['label']}]({src['url']})"
        trimmed = body.rstrip()
        terminal = _TERMINAL_RE.search(trimmed)
        if terminal:
            return trimmed[: terminal.start()] + " " + citation + trimmed[terminal.start():]
        return trimmed + " " + citation

    def _build_corpus(
        self, src_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build the deduped, tokenized source corpus from flat source dicts."""
        corpus: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for src in src_list:
            url = src.get("url")
            if not url:
                continue
            norm = self._normalize_url(url)
            if norm in seen:
                continue
            seen.add(norm)
            parts = [src.get("title") or "", src.get("text") or ""]
            highlights = src.get("highlights")
            if isinstance(highlights, list):
                parts.append(" ".join(str(h) for h in highlights))
            corpus.append(
                {
                    "label": self._label_for(src),
                    "url": url,
                    "tokens": self._tokenize(" ".join(parts)),
                }
            )
        return corpus

    def _label_for(self, src: Dict[str, Any]) -> str:
        """A short citation label drawn from the source title."""
        title = re.sub(r"\s+", " ", (src.get("title") or "").strip())
        if not title:
            return "Source"
        if len(title) > self.max_label_chars:
            title = title[: self.max_label_chars - 1].rstrip() + "…"
        return title

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

    def _is_structural(self, stripped: str) -> bool:
        """True for markdown lines that never carry citable prose."""
        return (
            stripped.startswith(("#", "|", "<"))
            or stripped in ("---", "***", "___")
        )

    def _peel_prefix(self, line: str) -> Tuple[str, str]:
        """Split a leading list/blockquote marker so prose can be cited safely."""
        match = _PREFIX_RE.match(line)
        if match:
            return match.group(1), match.group(2)
        return "", line

    def _urls_present(self, text: str) -> Set[str]:
        """Normalized URLs already referenced in ``text`` (linked or bare)."""
        return {
            self._normalize_url(m.group(0)) for m in _ANY_URL_RE.finditer(text)
        }

    def _tokenize(self, text: str) -> Set[str]:
        return set(_TOKEN_RE.findall(text.lower()))

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for set comparison (drop scheme/www/fragment)."""
        cleaned = url.strip().lower().split("#")[0]
        cleaned = re.sub(r"^https?://(www\.)?", "", cleaned)
        return cleaned.rstrip("/")
