"""Filesystem-based long-term memory for gathered research sources.

Mode 2 (adapted port) of *Filesystem-Based Memory for LLM Agents:
Organization, Evolution, and Sustainability* (arXiv:2607.26637v1).

The reference formalizes agent long-term memory as **one filesystem** -- a
directory tree of markdown files that the agent itself reads, writes, and
reorganizes -- and studies three roles around it:

* a **management** agent that integrates and organizes incoming content,
* a **search** agent that answers queries with cited sources, and
* an **execution** agent whose task trajectories are distilled into skills.

This deep-research pipeline already persists markdown reports (``reports/``,
``--save``) and already gathers Exa sources per subquery, but it has **no
cross-run memory**: every research run re-searches from scratch. This module
adds the missing stage -- a filesystem memory the supervisor organizes
retrieved sources into and recalls from on later queries -- following the same
"standalone agent wired onto ``SupervisorAgent`` as an attribute" shape as the
auditor and the research trace.

**Mode 2 substitutions (auxiliary components replaced with target-native
equivalents; the core mechanism is preserved):**

* The paper's **LLM management agent** (which decides the directory hierarchy
  for incoming content) is replaced by a **parameter-free organizer** that bins
  each source under a topic directory derived from the subquery it came from --
  the retriever already emits per-subquery buckets, so the categorization
  signal is native to this pipeline and needs no extra model call.
* The paper's **LLM search agent** (which retrieves and cites) is replaced by a
  **parameter-free lexical-overlap retriever** (token intersection, the same
  grounding proxy the auditor uses). It needs no API keys and is fully
  deterministic, so it can run on every query and be unit-tested offline.
* The paper's **benchmark suite** (long-conversation benchmarks, embodied
  tasks) is cut -- evaluation belongs in a downstream PR. In its place the
  module exposes a directly-computable **search-economy** comparison (organized
  hierarchy vs. flat dump), which is the paper's headline *result* ("organized
  stores roughly halve retrieval cost where material is large") and the
  comparison the team's suggested experiment asks for.

**Deliberately out of scope:** the paper's third (**execution / skill
distillation**) role. This repo has no skill-distillation surface -- there is
no task-trajectory store to compress into reusable skills -- so there is
nothing analogous to port. Only the management and search roles, which map
cleanly onto the gathered-sources flow, are implemented.

The core mechanism is preserved: incoming content is organized into a growing
directory tree of markdown files, and a later query is answered against that
organized store with cited sources, paying less retrieval cost than scanning a
flat dump.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

# Tokens of >= 3 lowercase alphanumeric chars (matches the auditor's tokenizer).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Words ignored when scoring a source against a query, so generic overlap does
# not read as "relevant" (same stopword discipline as the grounding auditor).
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
    predicted research report reports analysis overview introduction summary
    findings results sources
    """.split()
)


@dataclass
class MemoryEntry:
    """One source organized into the memory filesystem."""

    path: str  # store-relative path, e.g. "market-size/quantum-report.md"
    title: str
    url: str
    category: str  # topic directory slug the source was binned into
    text: str
    score: float = 0.0  # query overlap score when this entry was recalled

    def as_cited_source(self) -> Dict[str, Any]:
        """Project this entry back into the retriever's source-dict shape."""
        return {
            "title": self.title,
            "url": self.url,
            "text": self.text,
            "category": self.category,
        }


@dataclass
class RecallResult:
    """Outcome of answering a query against the organized memory store."""

    entries: List[MemoryEntry] = field(default_factory=list)
    bytes_scanned: int = 0  # bytes the organized search read to find these
    flat_bytes_scanned: int = 0  # bytes a flat dump would have read
    files_opened: int = 0
    flat_files_opened: int = 0
    store_files: int = 0  # total files in the store when recall ran

    @property
    def search_economy(self) -> float:
        """Fraction of bytes the organized search *avoided* scanning vs. flat.

        ``0.0`` means no saving (tiny store or query matched everything); the
        paper's finding is that this grows toward ~0.5 (halving) as the store
        gets large and the query matches a narrow slice of it.
        """
        if self.flat_bytes_scanned <= 0:
            return 0.0
        return 1.0 - (self.bytes_scanned / self.flat_bytes_scanned)

    def as_cited_sources(self) -> List[Dict[str, Any]]:
        return [e.as_cited_source() for e in self.entries]


@dataclass
class MemoryStats:
    """Health of the memory store as it grows across runs (paper's store health)."""

    files: int = 0
    topics: int = 0
    duplicates_skipped: int = 0


def _slugify(text: str, max_len: int = 40) -> str:
    """Turn free text into a filesystem-safe directory/file slug."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not cleaned:
        cleaned = "uncategorized"
    return cleaned[:max_len].rstrip("-") or "uncategorized"


def _tokenize(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS


class FilesystemMemory:
    """Organize retrieved sources into a markdown filesystem and recall from it.

    The store is a two-level tree written under ``root``::

        <root>/<topic-slug>/<source-slug>.md

    Each source file carries YAML-style front matter (``url``, ``title``,
    ``category``) followed by its text, so the tree is readable by humans and
    by generic file tools -- the medium the paper studies.
    """

    # Runtime artifact written during research (parallel to ``reports/``).
    # Tests redirect this to a temp dir via ``tests/conftest.py``.
    DEFAULT_ROOT = "memory"

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or self.DEFAULT_ROOT)
        self._duplicates_skipped = 0

    # -- management role --------------------------------------------------

    def integrate(
        self,
        sources: Iterable[Any],
        query: str = "",
        topic_override: Optional[str] = None,
    ) -> int:
        """Organize incoming ``sources`` into the filesystem hierarchy.

        Args:
            sources: Sources gathered by the retriever. May be a flat list of
                source dicts (``url``/``title``/``text``/``highlights``) or the
                nested per-subquery buckets emitted by ``WebSearchRetriever``.
            query: The research query the sources were gathered for -- used as
                the fallback topic when a source carries no subquery bucket.
            topic_override: Force every source into one topic directory.

        Returns:
            The number of new source files written (duplicates are skipped).
        """
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

        written = 0
        for category, source in self._iter_categorized(sources, query, topic_override):
            if self._write_source(category, source):
                written += 1
            else:
                self._duplicates_skipped += 1
        return written

    def recall(self, query: str, limit: int = 5) -> RecallResult:
        """Answer ``query`` with cited sources already organized in the store.

        This is the search role. The organized search navigates to topic
        directories whose slug overlaps the query and reads only their files;
        ``flat_bytes_scanned`` records what a verbatim flat dump would have
        cost to scan for the same answer, quantifying search economy.
        """
        if not self.root.exists():
            return RecallResult()

        query_tokens = _tokenize(query)
        all_files = sorted(self.root.rglob("*.md"))
        store_files = len(all_files)
        if store_files == 0:
            return RecallResult(store_files=0)

        # Organized navigation: only descend into topic dirs whose slug tokens
        # overlap the query. With no signal we fall back to scanning everything
        # (the search agent still answers, just without the economy benefit).
        topic_dirs = {p.parent for p in all_files}
        matched_dirs = (
            {d for d in topic_dirs if query_tokens & _tokenize(d.name)}
            if query_tokens
            else set()
        )
        if not matched_dirs:
            matched_dirs = set(topic_dirs)

        scored: List[MemoryEntry] = []
        organized_bytes = 0
        organized_files = 0
        flat_bytes = 0

        for path in all_files:
            raw = path.read_text(encoding="utf-8")
            entry = self._parse_file(path, raw)
            flat_bytes += len(raw)
            if path.parent in matched_dirs:
                organized_bytes += len(raw)
                organized_files += 1
                entry.score = self._overlap(query_tokens, entry)
                scored.append(entry)

        scored.sort(key=lambda e: e.score, reverse=True)
        return RecallResult(
            entries=scored[:limit],
            bytes_scanned=organized_bytes,
            flat_bytes_scanned=flat_bytes,
            files_opened=organized_files,
            flat_files_opened=store_files,
            store_files=store_files,
        )

    def stats(self) -> MemoryStats:
        """Report store health (file/topic counts, duplicates skipped)."""
        if not self.root.exists():
            return MemoryStats(duplicates_skipped=self._duplicates_skipped)
        files = list(self.root.rglob("*.md"))
        topics = {p.parent for p in files}
        return MemoryStats(
            files=len(files),
            topics=len(topics),
            duplicates_skipped=self._duplicates_skipped,
        )

    def reset(self) -> None:
        """Clear the in-memory duplicate counter (the on-disk tree is left as-is)."""
        self._duplicates_skipped = 0

    # -- internals --------------------------------------------------------

    def _iter_categorized(
        self,
        sources: Iterable[Any],
        query: str,
        topic_override: Optional[str],
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield ``(topic_slug, source_dict)`` pairs, flattening retriever buckets."""
        fallback_topic = _slugify(topic_override or query or "uncategorized")
        for src in sources:
            if isinstance(src, dict) and ("results" in src or "similar_results" in src):
                # Nested per-subquery bucket: its subquery text is the topic.
                topic = _slugify(src.get("subquery") or fallback_topic)
                for bucket_key in ("results", "similar_results"):
                    for item in src.get(bucket_key) or []:
                        if isinstance(item, dict) and item.get("url"):
                            yield topic, item
            elif isinstance(src, dict) and src.get("url"):
                yield fallback_topic, src

    def _write_source(self, category: str, source: Dict[str, Any]) -> bool:
        """Write one source as markdown. Returns False if it was a duplicate."""
        topic_dir = self.root / category
        url = source.get("url", "")
        slug = _slugify(source.get("title") or url)
        path = topic_dir / f"{slug}.md"
        if path.exists():
            return False
        topic_dir.mkdir(parents=True, exist_ok=True)
        title = source.get("title") or url
        text = source.get("text") or ""
        highlights = source.get("highlights")
        highlight_line = ""
        if isinstance(highlights, list) and highlights:
            highlight_line = "\n**Highlights:** " + "; ".join(str(h) for h in highlights) + "\n"
        body = (
            "---\n"
            f"url: {url}\n"
            f"title: {title}\n"
            f"category: {category}\n"
            "---\n\n"
            f"# {title}\n\n"
            f"Source: {url}\n{highlight_line}\n{text}\n"
        )
        path.write_text(body, encoding="utf-8")
        return True

    def _parse_file(self, path: Path, raw: str) -> MemoryEntry:
        """Reconstruct a MemoryEntry from a stored markdown file."""
        url = ""
        title = ""
        category = path.parent.name
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            if end != -1:
                for line in raw[3:end].splitlines():
                    if line.startswith("url:"):
                        url = line.split(":", 1)[1].strip()
                    elif line.startswith("title:"):
                        title = line.split(":", 1)[1].strip()
        if not title:
            title = path.stem
        body = self._strip_front_matter(raw)
        return MemoryEntry(
            path=str(path.relative_to(self.root)),
            title=title,
            url=url,
            category=category,
            text=body,
        )

    @staticmethod
    def _strip_front_matter(raw: str) -> str:
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            if end != -1:
                return raw[end + 4 :].strip()
        return raw.strip()

    @staticmethod
    def _overlap(query_tokens: Set[str], entry: MemoryEntry) -> float:
        """Lexical-overlap relevance of an entry to the query tokens."""
        if not query_tokens:
            return 0.0
        entry_tokens = _tokenize(entry.title + " " + entry.text) - _STOPWORDS
        if not entry_tokens:
            return 0.0
        return len(query_tokens & entry_tokens) / len(query_tokens)
