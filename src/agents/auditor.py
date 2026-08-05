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

**Scope of the port -- only Dimension 1 is ported.** BrainPilot's Auditor
persona audits *two* dimensions: (1) *evidence backing / fabrication* -- every
numeric claim, file/artifact reference, and external citation must trace to
evidence the session produced; and (2) *scientific reliability* -- the pipeline
behind a result-bearing claim is inspected for validity defects (data/label
leakage, metric misuse, baseline/chance confusion, double-dipping, etc.). This
port implements **only Dimension 1 (grounding/fabrication)**. Dimension 2 is
deliberately **not** ported: it inspects experiment-pipeline artifacts (split
logic, configs, logged metrics, model outputs) that exist in BrainPilot's
per-session workspace but that a general web deep-research tool never produces,
so there is nothing analogous to audit here.

Within Dimension 1, note the numeric check is a **lexical-overlap proxy**, not
the reference's exact-value verification: BrainPilot greps the workspace to
confirm each specific number appears in an artifact, whereas this port only
checks that a numeric claim's key terms overlap some retrieved source's text.

The core mechanism is preserved: an independent post-synthesis pass that links
each report citation and claim to retrieved evidence and flags the ones with no
support. Since LedgerMind (arXiv:2607.28374v1), that linking runs through a
:class:`~src.agents.evidence_ledger.EvidenceLedger` -- retrieved sources become
stable-id ledger entries, and each audited claim records the specific
entry(ies) that back it as provenance (``AuditResult.claim_provenance``). The
auditor remains the call site: it builds the ledger, resolves each citation and
claim to its backing entry, and renders the provenance into the report.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Tuple

from src.agents.evidence_ledger import EvidenceLedger, iter_flat_sources


# Markdown inline citation: [label](url)
_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")

# A sentence worth auditing if it carries a number (statistic / year / quantity).
_DIGIT_RE = re.compile(r"\d")


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
    # Structured Evidence Ledger (LedgerMind, arXiv:2607.28374v1): each
    # grounded claim is linked to the specific ledger entry(ies) backing it
    # as ``(claim, [(entry_id, title), ...])`` provenance.
    grounded_claims: int = 0
    claim_provenance: List[Tuple[str, List[Tuple[str, str]]]] = field(
        default_factory=list
    )
    cited_entry_ids: List[str] = field(default_factory=list)
    ledger_size: int = 0

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
            An :class:`AuditResult` summarizing grounding, with each audited
            claim linked (``claim_provenance``) to the ledger entry(ies) that
            back it.
        """
        src_list = list(iter_flat_sources(sources))
        if not src_list:
            # Nothing was retrieved to verify against -- do not penalize.
            return AuditResult(verifiable=False, score=1.0, sources_checked=0)

        ledger = EvidenceLedger(src_list, min_token_overlap=self.min_token_overlap)

        citations = self._extract_citations(report)
        cited_urls = [url for _, url in citations]
        total = len(citations)
        unsupported = [url for url in cited_urls if ledger.find_url(url) is None]
        grounded = total - len(unsupported)
        score = (grounded / total) if total else 1.0

        # Link each audited claim to the specific ledger entry(ies) backing it
        # (LedgerMind provenance). Claims with no backing entry are unsupported.
        claim_provenance: List[Tuple[str, List[Tuple[str, str]]]] = []
        unsupported_claims: List[str] = []
        for claim in self._extract_claim_sentences(report):
            backing = ledger.backing_entries(claim)
            if backing:
                claim_provenance.append(
                    (claim, [(entry.entry_id, entry.title) for entry in backing])
                )
            else:
                unsupported_claims.append(claim)

        return AuditResult(
            total_citations=total,
            grounded_citations=grounded,
            unsupported_citations=unsupported,
            unsupported_claims=unsupported_claims,
            sources_checked=ledger.size,
            verifiable=True,
            score=score,
            grounded_claims=len(claim_provenance),
            claim_provenance=claim_provenance,
            cited_entry_ids=sorted(ledger.active_entry_ids(cited_urls)),
            ledger_size=ledger.size,
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

        lines.append("### Claim provenance (Structured Evidence Ledger)")
        lines.append(
            "> Each audited claim is linked to the specific retrieved source(s)"
        )
        lines.append(
            "> that back it. Adapted from LedgerMind's Structured Evidence"
        )
        lines.append("> Ledger (arXiv:2607.28374v1).")
        lines.append("")
        lines.append(
            f"**Ledger:** {result.ledger_size} gathered source(s); "
            f"{len(result.cited_entry_ids)} cited in the report "
            f"({result.grounded_claims} grounded claim(s))."
        )
        if result.claim_provenance:
            for claim, entries in result.claim_provenance[:10]:
                snippet = claim.strip()[: self.max_claim_chars]
                cites = ", ".join(f"{eid} ({title})" for eid, title in entries)
                lines.append(f'- "{snippet}" — backed by {cites}')
        else:
            lines.append("- _No audited claims carried a statistic to trace._")
        lines.append("")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

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
