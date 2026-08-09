"""Typed evidence graph + structural validation of the claim-evidence chain.

Adapted from **EviGraph: Evidence-Guided Autonomous Research Agents**
(arXiv:2608.04738v1). EviGraph represents the research process as a *typed
evidence graph* (Problem / Gap / Hypothesis / Experiment / Finding / Claim)
that is the agent's operational state, and inspects the graph's evidence
chains for **missing dependencies**, **semantic misalignment**, and
**result-claim inconsistencies**; it localizes the *earliest* weak node and
checkpoints validated state so an unsuccessful repair cannot corrupt
previously validated evidence.

This is a **Mode 2 (adapted port)** of that mechanism for this web
deep-research pipeline, sitting alongside :class:`~src.agents.auditor.ReportAuditor`
(flat URL + lexical grounding) and :class:`~src.agents.research_trace.ResearchTrace`
(descriptive post-hoc record), which together leave a structural gap: nothing
validates that the report's claims form a complete, consistent evidence chain.

Substitutions (auxiliary components replaced with target-native equivalents):

* EviGraph's six node types describe a full hypothesis -> experiment -> finding
  research pipeline. A web deep-research run produces no experiments, so the
  typed graph is built from this pipeline's *actual* artifacts -- the research
  query (**Problem**), the sources gathered by the retriever (**Evidence**),
  and the report's inline-cited claims (**Claim**). The Problem -> Evidence ->
  Claim typed chain is the structural analogue; Gap / Hypothesis / Experiment
  are dropped because a web report has nothing to map them to.
* EviGraph's semantic-misalignment check uses an LLM judge. It is replaced by a
  **parameter-free lexical-overlap proxy** between a claim and the tokens of
  the evidence it cites (falling back to the whole evidence corpus when a claim
  cites nothing) -- no extra API keys, fully deterministic, runnable on every
  report and unit-testable offline. This is the same substitution the repo's
  ``ReportAuditor`` already makes for grounding.

Intentionally scoped out (not ported):

* EviGraph *regenerates* the downstream subgraph of a weak node and re-runs the
  agent. This pipeline has no subgraph-regeneration loop, so regeneration is
  **not** ported: this module delivers the *diagnostic* half -- it builds the
  typed graph, runs the three structural checks, localizes the earliest weak
  node, exposes a checkpoint/restore so a future repair pass cannot corrupt
  validated evidence, and surfaces a **Claim Support Rate** (EviGraph's
  headline metric). Wiring an actual regeneration loop needs a different
  supervisor architecture and is left to a downstream PR.

What is preserved is the core mechanism: an explicit typed evidence state that
is inspected for structural defects before the report is trusted, with the
earliest defect localized and the validated state checkpointed.
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

# Markdown inline citation: [label](url). group(1) is the label, group(2) the url.
_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")

# A sentence is treated as a claim if it carries a number (statistic / year / quantity).
_DIGIT_RE = re.compile(r"\d")

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Common words ignored when comparing a claim against evidence, so that
# superficial keyword overlap does not read as "aligned".
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
    predicted com https www http
    """.split()
)

# Typed node kinds -- EviGraph's Problem / Finding / Claim adapted to web research.
PROBLEM = "problem"
EVIDENCE = "evidence"
CLAIM = "claim"

_KIND_LABELS: Dict[str, str] = {
    PROBLEM: "Problem",
    EVIDENCE: "Evidence",
    CLAIM: "Claim",
}

# Structural defect categories (the three checks EviGraph runs over evidence chains).
MISSING_DEPENDENCY = "missing_dependency"
RESULT_CLAIM_INCONSISTENCY = "result_claim_inconsistency"
SEMANTIC_MISALIGNMENT = "semantic_misalignment"

_DEFECT_LABELS: Dict[str, str] = {
    MISSING_DEPENDENCY: "Missing dependency",
    RESULT_CLAIM_INCONSISTENCY: "Result-claim inconsistency",
    SEMANTIC_MISALIGNMENT: "Semantic misalignment",
}


@dataclass
class EvidenceNode:
    """A single node in the typed evidence graph."""

    node_id: int
    kind: str
    summary: str
    tokens: Set[str] = field(default_factory=set)
    url: Optional[str] = None  # normalized url of an evidence node
    cited_url: Optional[str] = None  # normalized url a claim attributes itself to
    support: Optional[int] = None  # evidence node_id a claim links to, if any
    parent: Optional[int] = None
    defect_categories: List[str] = field(default_factory=list)

    @property
    def is_weak(self) -> bool:
        """True when this node failed at least one structural check."""
        return bool(self.defect_categories)


@dataclass
class GraphValidation:
    """Outcome of a structural validation pass over the evidence graph."""

    total_claims: int = 0
    supported_claims: int = 0
    evidence_count: int = 0
    weak_nodes: List[EvidenceNode] = field(default_factory=list)
    earliest_weak_node: Optional[EvidenceNode] = None

    @property
    def support_rate(self) -> float:
        """Fraction of claims with no structural defect (EviGraph's support rate)."""
        return (self.supported_claims / self.total_claims) if self.total_claims else 1.0


class EvidenceGraph:
    """Typed evidence graph for one synthesized report.

    Built from the research query (Problem), the sources gathered by the
    retriever (Evidence), and the report's claims (Claim). ``validate`` runs
    EviGraph's three structural checks over each claim's evidence chain,
    localizes the earliest weak node, and the checkpoint/restore pair protects
    a previously validated state from a failed downstream repair.

    The graph is deliberately parameter-free: it consumes only the report and
    gathered sources the supervisor already has on hand, so it adds no API
    calls and runs deterministically offline.
    """

    def __init__(self, min_token_overlap: int = 3, max_claim_chars: int = 160):
        self.min_token_overlap = min_token_overlap
        self.max_claim_chars = max_claim_chars
        self._nodes: List[EvidenceNode] = []
        self._snapshot: Optional[List[EvidenceNode]] = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_run(
        cls,
        report: str,
        sources: Iterable[Any],
        query: str = "",
        **kwargs: Any,
    ) -> "EvidenceGraph":
        """Build the typed graph from a run's report, gathered sources, and query."""
        graph = cls(**kwargs)
        if query and query.strip():
            graph.add_problem(query)
        for src in cls._iter_sources(sources):
            graph.add_evidence(src)
        for text, url in cls._extract_claims(report):
            graph.add_claim(text, cited_url=url)
        return graph

    @property
    def nodes(self) -> List[EvidenceNode]:
        """Recorded nodes in insertion order (a copy, safe to iterate)."""
        return list(self._nodes)

    def node_counts(self) -> Dict[str, int]:
        """Count of nodes per typed kind."""
        counts = {PROBLEM: 0, EVIDENCE: 0, CLAIM: 0}
        for node in self._nodes:
            counts[node.kind] = counts.get(node.kind, 0) + 1
        return counts

    def add_problem(self, query: str) -> int:
        """Record the research question the report answers (EviGraph Problem node)."""
        return self._append(PROBLEM, query.strip())

    def add_evidence(self, source: Dict[str, Any]) -> int:
        """Record a gathered source as typed evidence the report may draw on."""
        url = self._normalize_url(source.get("url") or "") or None
        title = (source.get("title") or url or "source")
        node_id = self._append(
            EVIDENCE,
            self._truncate(str(title)),
            tokens=self._tokenize_source(source),
            url=url,
        )
        return node_id

    def add_claim(self, text: str, cited_url: Optional[str] = None) -> int:
        """Record a report claim and link it to the evidence it cites, if any."""
        norm = self._normalize_url(cited_url) if cited_url else None
        support = self._find_evidence(norm) if norm else None
        return self._append(
            CLAIM,
            self._truncate(text),
            tokens=self._tokenize(text),
            cited_url=norm,
            support=support,
            parent=support,
        )

    # -- validation --------------------------------------------------------

    def validate(self) -> GraphValidation:
        """Run EviGraph's three structural checks over every claim's evidence chain.

        Per claim:

        * **missing dependency** -- the claim links to no evidence node (it is
          orphaned from the evidence chain, or no evidence was gathered at all).
        * **result-claim inconsistency** -- the claim cites a specific URL that
          is not among the gathered evidence (its stated support does not exist).
        * **semantic misalignment** -- the claim's key terms do not overlap its
          cited evidence (or, when uncited, the whole corpus) enough to read as
          supported -- a lexical proxy for EviGraph's LLM misalignment judge.

        Returns the support rollup plus the earliest weak node in document order.
        """
        evidence_nodes = [n for n in self._nodes if n.kind == EVIDENCE]
        claim_nodes = [n for n in self._nodes if n.kind == CLAIM]
        has_evidence = bool(evidence_nodes)
        corpus_tokens: Set[str] = set()
        for ev in evidence_nodes:
            corpus_tokens |= ev.tokens

        for claim in claim_nodes:
            categories: List[str] = []
            if claim.support is None:
                categories.append(MISSING_DEPENDENCY)
            if claim.cited_url is not None and claim.support is None:
                categories.append(RESULT_CLAIM_INCONSISTENCY)
            if has_evidence:
                ref_tokens = self._support_tokens(claim) or corpus_tokens
                claim_keys = claim.tokens - _STOPWORDS
                if ref_tokens and claim_keys and len(claim_keys & ref_tokens) < self.min_token_overlap:
                    categories.append(SEMANTIC_MISALIGNMENT)
            claim.defect_categories = categories

        weak = [c for c in claim_nodes if c.is_weak]
        return GraphValidation(
            total_claims=len(claim_nodes),
            supported_claims=len(claim_nodes) - len(weak),
            evidence_count=len(evidence_nodes),
            weak_nodes=weak,
            earliest_weak_node=weak[0] if weak else None,
        )

    # -- checkpointing -----------------------------------------------------

    def checkpoint(self) -> int:
        """Snapshot the current graph so a failed repair cannot corrupt it.

        Returns the number of nodes snapshotted. Mirrors EviGraph's graph
        checkpointing: validated evidence is preserved and can be restored if a
        downstream regeneration step makes things worse rather than better.
        """
        self._snapshot = copy.deepcopy(self._nodes)
        return len(self._snapshot)

    def restore(self) -> bool:
        """Revert the graph to the last checkpoint, discarding later changes."""
        if self._snapshot is None:
            return False
        self._nodes = copy.deepcopy(self._snapshot)
        return True

    # -- rendering ---------------------------------------------------------

    def format_report(self, result: GraphValidation) -> str:
        """Render the validation as a markdown section to append to a report."""
        lines: List[str] = ["", "", "---", "", "## Evidence Graph Validation", ""]
        lines.append("> Structural check of the report's claim-evidence chain:")
        lines.append("> every claim is linked to typed evidence and inspected for")
        lines.append("> missing dependencies, result-claim inconsistencies, and")
        lines.append("> semantic misalignment. Adapted from EviGraph (arXiv:2608.04738v1).")
        lines.append("")

        counts = self.node_counts()
        lines.append(
            f"**Typed graph:** {counts[PROBLEM]} problem, "
            f"{result.evidence_count} evidence, {result.total_claims} claim node(s)."
        )
        pct = round(result.support_rate * 100)
        lines.append(
            f"**Claim support rate:** {result.supported_claims}/{result.total_claims} "
            f"claims structurally grounded ({pct}%)."
        )
        lines.append("")

        lines.append("### Earliest weak node (repair focus)")
        node = result.earliest_weak_node
        if node is not None:
            labels = ", ".join(_DEFECT_LABELS[c] for c in node.defect_categories)
            lines.append(f"- **[{node.node_id}] Claim:** \"{self._clip(node.summary)}\" — {labels}")
        else:
            lines.append("- _None — every claim traces to typed evidence._")
        lines.append("")

        lines.append("### Flagged claims")
        if result.weak_nodes:
            for n in result.weak_nodes[:10]:
                labels = ", ".join(_DEFECT_LABELS[c] for c in n.defect_categories)
                lines.append(f"- **[{n.node_id}]** {labels}: \"{self._clip(n.summary)}\"")
        else:
            lines.append("- _None._")
        lines.append("")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _append(self, kind: str, summary: str, **fields: Any) -> int:
        node = EvidenceNode(node_id=len(self._nodes) + 1, kind=kind, summary=summary.strip(), **fields)
        self._nodes.append(node)
        return node.node_id

    def _find_evidence(self, url: Optional[str]) -> Optional[int]:
        if not url:
            return None
        for node in self._nodes:
            if node.kind == EVIDENCE and node.url == url:
                return node.node_id
        return None

    def _support_tokens(self, claim: EvidenceNode) -> Set[str]:
        """Tokens of the specific evidence a claim cites (empty if uncited)."""
        if claim.support is None:
            return set()
        for node in self._nodes:
            if node.node_id == claim.support:
                return node.tokens
        return set()

    @staticmethod
    def _iter_sources(sources: Iterable[Any]) -> Iterator[Dict[str, Any]]:
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

    @staticmethod
    def _extract_claims(report: str) -> List[Tuple[str, Optional[str]]]:
        """Return ``(sentence, first_cited_url_or_None)`` for each report claim.

        A sentence is a claim if it carries a number (statistic / year) or an
        inline citation; its first citation's URL (if any) is the evidence it
        attributes itself to.
        """
        kept: List[str] = []
        for line in report.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "|", "```")):
                continue
            kept.append(stripped)
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(kept))
        claims: List[Tuple[str, Optional[str]]] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            has_citation = bool(_LINK_RE.search(sentence))
            if not (has_citation or (_DIGIT_RE.search(sentence) and len(sentence) > 20)):
                continue
            match = _LINK_RE.search(sentence)
            url = match.group(2) if match else None
            claims.append((sentence, url))
        return claims

    def _tokenize_source(self, source: Dict[str, Any]) -> Set[str]:
        parts = [str(source.get("title") or ""), str(source.get("text") or "")]
        highlights = source.get("highlights")
        if isinstance(highlights, list):
            parts.append(" ".join(str(h) for h in highlights))
        return self._tokenize(" ".join(parts))

    def _tokenize(self, text: str) -> Set[str]:
        # Keep citation labels but drop their urls so urls do not pose as terms.
        cleaned = _LINK_RE.sub(lambda m: m.group(1), text.lower())
        return set(_TOKEN_RE.findall(cleaned))

    def _normalize_url(self, url: str) -> Optional[str]:
        cleaned = url.strip().lower().split("#")[0]
        cleaned = re.sub(r"^https?://(www\.)?", "", cleaned)
        cleaned = cleaned.rstrip("/")
        return cleaned or None

    def _truncate(self, text: str) -> str:
        text = " ".join(text.split())
        return text if len(text) <= self.max_claim_chars else text[: self.max_claim_chars - 1] + "…"

    def _clip(self, text: str) -> str:
        text = text.strip()
        return text if len(text) <= self.max_claim_chars else text[: self.max_claim_chars - 1] + "…"
