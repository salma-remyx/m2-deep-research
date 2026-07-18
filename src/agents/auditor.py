"""Post-synthesis grounding auditor for research reports.

Adapted from the **Auditor agent** in *BrainPilot: Automating Brain Discovery
with Agentic Research* (arXiv:2607.15079v1). BrainPilot's Auditor is an
independent agent that runs a fabrication-checking pass over the synthesized
output, linking every claim back to evidence before the result is trusted.

This is a **Mode 2 (adapted port)** of that mechanism for this deep-research
pipeline:

* BrainPilot's curated brain-science knowledge base (7,233 indexed items) is
  replaced by the **retrieved Exa sources** that already back this pipeline's
  reports -- i.e. the natural evidence corpus for a report produced by the
  ``WebSearchRetriever``.
* BrainPilot's LLM fabrication judge is replaced by a **parameter-free
  grounding proxy** (citation-URL matching + lexical claim overlap). It needs
  no extra API keys and is fully deterministic, so it can run on every report
  and be unit-tested offline.

The core mechanism is preserved: an independent post-synthesis pass that links
each report citation and claim to retrieved evidence and flags the ones with no
support -- the ``Graph of Trace`` idea, rendered as a per-claim evidence check.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple


# Markdown inline citation: [label](url)
_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")

# A sentence worth auditing if it carries a number (statistic / year / quantity).
_DIGIT_RE = re.compile(r"\d")

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Common words ignored when comparing a claim against source text, so that
# superficial keyword overlap does not read as "grounded".
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


@dataclass
class AuditResult:
    """Outcome of a grounding audit over a single report."""

    total_citations: int = 0
    grounded_citations: int = 0
    unsupported_citations: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    sources_checked: int = 0
    verifiable: bool = True
    score: float = 1.0  # fraction of citations backed by retrieved sources

    @property
    def passed(self) -> bool:
        """True when nothing ungrounded was found."""
        return self.verifiable and not self.unsupported_citations and not self.unsupported_claims


class ReportAuditor:
    """Verify a synthesized report is grounded in retrieved sources.

    The auditor is deliberately parameter-free: given a report (markdown with
    inline ``[text](url)`` citations) and the sources gathered by the
    retriever, it flags citations whose URL never appeared in the retrieved
    set (a strong fabrication signal) and numeric claims whose key terms have
    no lexical overlap with any source (a weaker, complementary signal).
    """

    def __init__(self, min_token_overlap: int = 3, max_claim_chars: int = 160):
        self.min_token_overlap = min_token_overlap
        self.max_claim_chars = max_claim_chars

    def audit(self, report: str, sources: Iterable[Any]) -> AuditResult:
        """Audit ``report`` against ``sources`` retrieved for it.

        Args:
            report: Synthesized report text (may contain markdown citations).
            sources: Retrieved sources. May be a flat list of source dicts
                (``url``/``title``/``text``/``highlights``) or the nested
                subquery buckets emitted by ``WebSearchRetriever``.

        Returns:
            An :class:`AuditResult` summarizing grounding.
        """
        src_list = list(self._iter_sources(sources))
        if not src_list:
            # Nothing was retrieved to verify against -- do not penalize.
            return AuditResult(verifiable=False, score=1.0, sources_checked=0)

        urls, texts = self._source_corpus(src_list)

        citations = self._extract_citations(report)
        total = len(citations)
        unsupported = [
            url for _, url in citations if self._normalize_url(url) not in urls
        ]
        grounded = total - len(unsupported)
        score = (grounded / total) if total else 1.0

        unsupported_claims = [
            claim
            for claim in self._extract_claim_sentences(report)
            if not self._claim_is_grounded(claim, texts)
        ]

        return AuditResult(
            total_citations=total,
            grounded_citations=grounded,
            unsupported_citations=unsupported,
            unsupported_claims=unsupported_claims,
            sources_checked=len(src_list),
            verifiable=True,
            score=score,
        )

    def format_report(self, result: AuditResult) -> str:
        """Render an audit result as a markdown section to append to a report."""
        lines: List[str] = ["", "", "---", "", "## Source Grounding Audit", ""]
        lines.append(
            "> Independent post-synthesis pass that checks the report's citations"
        )
        lines.append(
            "> and claims against retrieved sources. Adapted from BrainPilot's"
        )
        lines.append("> Auditor agent (arXiv:2607.15079v1).")
        lines.append("")

        if not result.verifiable:
            lines.append(
                "**Note:** no retrieved sources were captured for this report; "
                "grounding could not be verified."
            )
            return "\n".join(lines)

        if result.total_citations == 0:
            lines.append("**Citations:** no inline citations found to verify.")
        else:
            pct = round(result.score * 100)
            lines.append(
                f"**Citation grounding:** {result.grounded_citations} / "
                f"{result.total_citations} cited URLs match retrieved sources "
                f"({pct}%)."
            )
        lines.append("")

        lines.append("### Unsupported citations (potential fabrications)")
        if result.unsupported_citations:
            for url in result.unsupported_citations:
                lines.append(f"- `{url}` — not found among retrieved sources")
        else:
            lines.append("- _None — all citations trace to retrieved sources._")
        lines.append("")

        lines.append("### Unsupported claims (no lexical match in retrieved sources)")
        if result.unsupported_claims:
            for claim in result.unsupported_claims[:10]:
                snippet = claim.strip()[: self.max_claim_chars]
                lines.append(f'- "{snippet}"')
        else:
            lines.append("- _None._")
        lines.append("")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _iter_sources(self, sources: Iterable[Any]) -> Iterator[Dict[str, Any]]:
        """Yield flat source dicts, flattening retriever subquery buckets."""
        for src in sources:
            if not isinstance(src, dict):
                continue
            if "results" in src or "similar_results" in src:
                # Nested subquery bucket from WebSearchRetriever.
                for bucket_key in ("results", "similar_results"):
                    for item in src.get(bucket_key) or []:
                        if isinstance(item, dict):
                            yield item
            else:
                yield src

    def _source_corpus(self, src_list: List[Dict[str, Any]]) -> Tuple[Set[str], List[str]]:
        """Build the normalized URL set and text corpus from flat sources."""
        urls: Set[str] = set()
        texts: List[str] = []
        for src in src_list:
            url = src.get("url")
            if url:
                urls.add(self._normalize_url(url))
            parts = [src.get("title") or "", src.get("text") or ""]
            highlights = src.get("highlights")
            if isinstance(highlights, list):
                parts.append(" ".join(str(h) for h in highlights))
            texts.append(" ".join(parts))
        return urls, texts

    def _extract_citations(self, report: str) -> List[Tuple[str, str]]:
        """Return ``(label, url)`` for every markdown link in the report."""
        return _LINK_RE.findall(report)

    def _extract_claim_sentences(self, report: str) -> List[str]:
        """Return prose sentences carrying a number (heuristic claim detector)."""
        # Drop citation URLs (keep labels) so URLs do not fake sentence breaks.
        prose = _LINK_RE.sub(lambda m: m.group(1), report)
        kept: List[str] = []
        for line in prose.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip structural markdown: headers, table rows, code fences.
            if stripped.startswith(("#", "|", "```")):
                continue
            kept.append(stripped)
        joined = " ".join(kept)
        sentences = re.split(r"(?<=[.!?])\s+", joined)
        return [s.strip() for s in sentences if _DIGIT_RE.search(s) and len(s.strip()) > 20]

    def _claim_is_grounded(self, claim: str, source_texts: List[str]) -> bool:
        """True if the claim shares enough key terms with any source text."""
        claim_tokens = self._tokenize(claim) - _STOPWORDS
        if not claim_tokens:
            return True
        for text in source_texts:
            if len(claim_tokens & self._tokenize(text)) >= self.min_token_overlap:
                return True
        return False

    def _tokenize(self, text: str) -> Set[str]:
        return set(_TOKEN_RE.findall(text.lower()))

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for set comparison (drop scheme/www/fragment)."""
        cleaned = url.strip().lower().split("#")[0]
        cleaned = re.sub(r"^https?://(www\.)?", "", cleaned)
        return cleaned.rstrip("/")
