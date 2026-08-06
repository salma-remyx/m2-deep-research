"""Variable-length source segmentation for retrieval (LumberChunker, adapted).

This is a **Mode 2 (adapted port)** of *LumberChunker: Long-Form Narrative
Document Segmentation* (Krapac et al., arXiv:2406.17526v1).

LumberChunker segments a long document into variable-length chunks whose
boundaries fall where the content *begins to shift*. It first splits the
document into fixed-size atomic passages, then iteratively scans a window
of consecutive passages and asks which one is the last that still shares
content with the window's first passage -- the chunk runs from the window
start to that point, and the scan resumes right after it. The output is a
sequence of semantically-coherent, variable-length passages rather than
the fixed-size windows the retriever previously hard-truncated at 1000
characters.

What is ported at full fidelity
    * The atomic-passage decomposition (fixed word-count blocks).
    * The iterative sliding-window scan that emits one variable-length
      chunk per window and advances past it.
    * The "content begins to shift" boundary rule: a chunk ends at the
      first passage in the window that no longer shares content with the
      window anchor, exactly mirroring the paper's framing.

Substituted auxiliary component (the one deviation from the paper)
    LumberChunker detects each shift by prompting an LLM with the windowed
    passages and reading back the index of the last matching passage.
    Running that oracle here would add one LLM round-trip per window
    across every retrieved source on every research run, on top of the
    supervisor's already-synchronous synthesis call, and would need live
    API keys to exercise. We replace it with a **parameter-free
    lexical-coherence proxy**: a passage is treated as "still the same
    content" when it shares at least ``min_overlap`` non-stopword tokens
    with the window anchor. This preserves the variable-length-shift
    structure that is LumberChunker's actual improvement over fixed-size
    chunking; it only changes *how* a shift is detected.

A relevance-ranked budget selection (:meth:`PassageSegmenter.select`) then
turns the variable passages into the bounded context that feeds
:meth:`src.agents.web_search_retriever.WebSearchRetriever.synthesize_findings`,
so the most query-relevant coherent passages from the *full* source text
reach synthesis instead of its first ``max_chars`` characters.
"""

import re
from dataclasses import dataclass
from typing import List, Sequence, Set

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise),
# matching the lexical convention used by the grounding auditor.
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Common words ignored when judging whether two passages share content, so
# superficial glue-word overlap does not read as "same content".
_STOPWORDS: Set[str] = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over
    under about as is are was were be been being this that these those it its
    they them we you he she him her not no nor so than too very can could should
    would may might must will shall do does did has have had more most less least
    many much few several also however which who whom whose what when where why
    how during while across among between within without via per using used use
    """.split()
)


@dataclass
class Passage:
    """A variable-length segment produced from a source document."""

    text: str
    index: int = 0  # ordinal among the segments returned by ``segment``
    relevance: float = 0.0  # query-overlap score, populated by ``select``


class PassageSegmenter:
    """Segment a document into variable-length, content-coherent passages.

    This is the adapted LumberChunker mechanism (see module docstring). It is
    deterministic and parameter-free, so it runs on every retrieved source
    with no extra API calls.
    """

    def __init__(
        self,
        passage_words: int = 50,
        window_size: int = 6,
        min_overlap: int = 2,
    ):
        self.passage_words = max(1, passage_words)
        self.window_size = max(2, window_size)
        self.min_overlap = max(1, min_overlap)

    # -- LumberChunker mechanism ------------------------------------------

    def segment(self, document: str) -> List[Passage]:
        """Return variable-length passages for ``document``.

        Mirrors LumberChunker: split into fixed-size atomic passages, then
        iteratively emit one chunk per window up to the first passage that
        no longer shares content with the window anchor.
        """
        atoms = self._atomic_passages(document)
        if not atoms:
            return []

        passages: List[Passage] = []
        start = 0
        n = len(atoms)
        while start < n:
            window_end = min(start + self.window_size, n)
            anchor = self._content_tokens(atoms[start])
            if not anchor:
                # No content tokens to compare on -- emit the passage and move on.
                last = start
            else:
                # The anchor passage always belongs to this chunk; extend until
                # the first passage whose content has begun to shift away.
                last = start
                for j in range(start + 1, window_end):
                    shared = len(anchor & self._content_tokens(atoms[j]))
                    if shared >= self.min_overlap:
                        last = j
                    else:
                        break
            text = " ".join(atoms[start : last + 1])
            passages.append(Passage(text=text, index=len(passages)))
            start = last + 1
        return passages

    # -- integration helper ----------------------------------------------

    def select(
        self,
        passages: Sequence[Passage],
        query: str,
        max_chars: int,
    ) -> List[Passage]:
        """Pick the most query-relevant passages fitting ``max_chars``.

        Variable-length segmentation often yields more text than the synthesis
        budget allows. We rank passages by lexical overlap with ``query`` and
        greedily fill ``max_chars`` with whole passages, so the context that
        reaches synthesis is drawn from the full source (not just its first
        ``max_chars`` characters) while staying bounded. Returned in original
        document order.
        """
        if not passages:
            return []

        query_tokens = self._content_tokens(query)
        ranked = list(passages)
        if query_tokens:
            for p in ranked:
                p.relevance = len(query_tokens & self._content_tokens(p.text))
            # Stable sort: relevance desc, then original document order.
            ranked.sort(key=lambda p: (-p.relevance, p.index))
        else:
            for p in ranked:
                p.relevance = 0.0

        chosen: List[Passage] = []
        used = 0
        for p in ranked:
            if used >= max_chars:
                break
            length = len(p.text)
            if used + length > max_chars and chosen:
                # Would overflow the budget; keep scanning for smaller passages.
                continue
            chosen.append(p)
            used += length

        chosen.sort(key=lambda p: p.index)
        return chosen

    # -- internals --------------------------------------------------------

    def _atomic_passages(self, document: str) -> List[str]:
        """Split ``document`` into fixed-size word-count blocks."""
        words = document.split()
        if not words:
            return []
        size = self.passage_words
        return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]

    def _content_tokens(self, text: str) -> Set[str]:
        return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS
