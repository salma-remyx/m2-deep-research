"""Pre-synthesis evidence-state closure tracker.

Adapted from the **evidence-state system** in *Omni-Decision: A Progressive
Evidence-State Agent System for Omni-Modal QA* (arXiv:2607.11433v1).
Omni-Decision maintains, for each query, a structured evidence state with
*confirmed evidence*, *unresolved conflicts*, and *open evidence needs*, and
uses that shared view to drive targeted evidence acquisition, repair, and a
stopping decision.

This is a **Mode 2 (adapted port)** of that mechanism for this deep-research
pipeline:

* Omni-Decision's heterogeneous observation normalizers (media / web /
  computation / verification modules) are replaced by this pipeline's
  **web-only retrieved source buckets** -- already normalized by the
  :class:`~src.agents.web_search_retriever.WebSearchRetriever` and structured
  one bucket per planned subquery.
* Omni-Decision's LLM evidence judge (which decides whether an observation
  confirms, contradicts, or leaves open a need) is replaced by a
  **parameter-free proxy** matching this repo's house style: lexical
  corroboration (independent sources sharing salient terms) signals
  *confirmed*; corroborating sources that cite no shared figure signal
  *conflict*; a need with too few or non-corroborating sources stays *open*.
  It is fully deterministic and needs no extra API calls, so it can run on
  every report and be unit-tested offline -- the same trade-off the
  :class:`~src.agents.auditor.ReportAuditor` makes for its grounding check.

**Scope of the port -- the control loop is intentionally not ported.**
Omni-Decision's contribution has two halves: (1) the *structured evidence
state* itself, and (2) a *control loop* that re-conditions planning,
acquisition, validation, repair, and stopping off that state. This port
implements **only half (1)** -- it maintains the per-need state during the run
and renders a final closure snapshot (confirmed / conflicting / open needs
plus a closure ratio) appended to the report. Half (2), the multi-round
repair/stopping loop that feeds the state back into the supervisor to drive
targeted re-search, is deliberately **not** ported: this pipeline's tool-use
loop is single-shot and driven by the Minimax supervisor's own interleaved
thinking, so a separate repair planner is a downstream concern. The closure
snapshot this port renders is exactly the signal such a loop would consume.

The tracker complements the post-synthesis grounding auditor (*are the
report's claims backed?*) and the Graph of Trace (*how was the report
built?*) by answering a third question: *which planned information needs are
confirmed, which conflict, and which are still open?*
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Need statuses, ordered from least to most closed. "open" covers both needs
# with no evidence and needs whose evidence is too thin to corroborate.
CONFIRMED = "confirmed"
CONFLICTING = "conflicting"
OPEN = "open"

# Markdown bullet marker per status, so the rendered snapshot reads at a glance.
_STATUS_MARKER: Dict[str, str] = {
    CONFIRMED: "✓",
    CONFLICTING: "⚠",
    OPEN: "○",
}

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise),
# matching the auditor's tokenization convention.
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Numeric quantities in source text (percents, plain ints/floats).
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")

# Common words ignored when comparing sources, so shallow keyword overlap does
# not read as corroboration. Compact and self-contained on purpose.
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
    predicted reach reaches reaching projected according billion million percent
    """.split()
)


@dataclass
class EvidenceNeed:
    """One open information need (a planned subquery) and its gathered evidence."""

    query: str
    status: str = OPEN
    sources: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        """Number of distinct sources committed for this need."""
        return len(self.sources)


@dataclass
class EvidenceState:
    """Snapshot of evidence closure across every planned need."""

    needs: List[EvidenceNeed] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.needs)

    @property
    def confirmed(self) -> List[EvidenceNeed]:
        return [n for n in self.needs if n.status == CONFIRMED]

    @property
    def conflicting(self) -> List[EvidenceNeed]:
        return [n for n in self.needs if n.status == CONFLICTING]

    @property
    def open_needs(self) -> List[EvidenceNeed]:
        return [n for n in self.needs if n.status == OPEN]

    @property
    def closure_ratio(self) -> float:
        """Fraction of needs with corroborated, non-conflicting evidence."""
        if not self.total:
            return 1.0
        return len(self.confirmed) / self.total


class EvidenceStateTracker:
    """Maintain the per-need evidence state for one research run.

    Parameter-free and deterministic. The supervisor registers the planned
    subqueries as open needs, commits each retrieved source bucket as evidence
    against its need, then asks the tracker to classify the state and render a
    closure snapshot. Needs are matched to buckets by normalized subquery text,
    and any bucket without a registered need is added on the fly, so the tracker
    stays correct even when planning was not recorded.
    """

    def __init__(self, confirm_threshold: int = 2, min_overlap: int = 3):
        """
        Args:
            confirm_threshold: Independent sources that must corroborate a need
                (share salient terms) before it counts as confirmed.
            min_overlap: Salient terms two sources must share to count as
                corroboration (mirrors the auditor's token-overlap floor).
        """
        self.confirm_threshold = confirm_threshold
        self.min_overlap = min_overlap
        self._needs: Dict[str, EvidenceNeed] = {}
        self._order: List[str] = []

    @property
    def needs(self) -> List[EvidenceNeed]:
        """Registered needs in insertion (planning) order."""
        return [self._needs[k] for k in self._order]

    def reset(self) -> None:
        """Clear the state so the tracker can be reused for a new run."""
        self._needs = {}
        self._order = []

    def register_needs_from_plan(self, plan_result: str) -> List[EvidenceNeed]:
        """Parse a ``planning_agent`` result and register its subqueries as needs.

        ``plan_result`` is the JSON string the planning agent returns (a
        ``{"subqueries": [{"query": ...}, ...]}`` envelope). Unparseable input
        is ignored -- the tracker then falls back to registering needs from the
        retriever buckets themselves.
        """
        for subquery in self._parse_subqueries(plan_result):
            self._register(subquery.get("query", ""))
        return self.needs

    def commit_evidence(self, sources: Iterable[Any]) -> None:
        """Commit retrieved source buckets against their matching needs.

        ``sources`` is the ``WebSearchRetriever`` output: a list of per-subquery
        buckets (``subquery`` / ``results`` / ``similar_results``). A flat list
        of sources with no subquery info collapses into a single unlabeled need.
        """
        for subquery, srcs in self._iter_buckets(sources):
            need = self._register(subquery)
            need.sources.extend(srcs)

    def classify(self) -> EvidenceState:
        """Compute each need's status from its committed sources, in place.

        Returns the state snapshot with needs in insertion order.
        """
        for need in self._needs.values():
            need.status, need.conflicts = self._classify_need(need)
        return EvidenceState(needs=self.needs)

    def render(self, state: Optional[EvidenceState] = None) -> str:
        """Render the closure snapshot as a markdown ``## Evidence State`` section."""
        if state is None:
            state = self.classify()

        lines: List[str] = ["", "", "---", "", "## Evidence State", ""]
        lines.append("> Closure snapshot for each planned subquery (an open")
        lines.append("> information need): whether the gathered evidence")
        lines.append("> confirms it, conflicts, or leaves it open. Adapted from")
        lines.append("> Omni-Decision's evidence-state system (arXiv:2607.11433).")
        lines.append("")

        if not state.needs:
            lines.append("_No evidence needs were recorded for this report._")
            return "\n".join(lines)

        pct = round(state.closure_ratio * 100)
        lines.append(
            f"**Evidence closure:** {len(state.confirmed)}/{state.total} needs "
            f"confirmed ({pct}%), {len(state.conflicting)} conflicting, "
            f"{len(state.open_needs)} open."
        )
        lines.append("")

        lines.append("### Per-need status")
        for need in state.needs:
            marker = _STATUS_MARKER.get(need.status, "•")
            lines.append(
                f"- {marker} **{need.status.title()}:** {need.query} "
                f"_({need.source_count} source(s))_"
            )
        lines.append("")

        if state.open_needs:
            lines.append("### Open evidence needs (targeted re-search candidates)")
            for need in state.open_needs:
                if need.source_count:
                    note = "below corroboration threshold"
                else:
                    note = "no sources gathered"
                lines.append(f"- {need.query} — {note}")
            lines.append("")

        if state.conflicting:
            lines.append("### Conflicting evidence (unresolved)")
            for need in state.conflicting:
                for conflict in need.conflicts:
                    lines.append(f"- {need.query} — {conflict}")
            lines.append("")

        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _register(self, query: str) -> EvidenceNeed:
        key = self._normalize_query(query)
        if not key:
            key = "(empty subquery)"
        if key not in self._needs:
            self._needs[key] = EvidenceNeed(query=(query.strip() or key))
            self._order.append(key)
        return self._needs[key]

    def _classify_need(self, need: EvidenceNeed) -> Tuple[str, List[str]]:
        """Classify one need from its sources.

        * Fewer sources than the corroboration threshold -> OPEN (too thin).
        * Sources that share no salient terms -> OPEN (fragmented, not about the
          same thing).
        * Corroborating sources that cite no shared figure -> CONFLICTING
          (topical agreement but divergent numbers).
        * Otherwise -> CONFIRMED.
        """
        sources = need.sources
        if len(sources) < self.confirm_threshold:
            return OPEN, []

        token_sets = [self._tokenize(self._source_text(s)) for s in sources]
        number_sets = [self._numbers(self._source_text(s)) for s in sources]

        corroborating = self._corroborating_indices(token_sets)
        if not corroborating:
            return OPEN, []

        citing = [number_sets[i] for i in corroborating if number_sets[i]]
        if len(citing) >= 2:
            shared = set.intersection(*citing)
            if not shared:
                divergent = sorted(set().union(*citing))
                return CONFLICTING, [
                    "corroborating sources cite no shared figure "
                    f"(divergent values: {', '.join(divergent)})"
                ]
        return CONFIRMED, []

    def _corroborating_indices(self, token_sets: List[Set[str]]) -> List[int]:
        """Indices of sources that share >= ``min_overlap`` terms with another."""
        corroborating: Set[int] = set()
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                if len(token_sets[i] & token_sets[j]) >= self.min_overlap:
                    corroborating.add(i)
                    corroborating.add(j)
        return sorted(corroborating)

    @staticmethod
    def _iter_buckets(
        sources: Iterable[Any],
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """Group flat sources or retriever buckets by subquery text."""
        buckets: List[Tuple[str, List[Dict[str, Any]]]] = []
        flat: List[Dict[str, Any]] = []
        for src in sources:
            if not isinstance(src, dict):
                continue
            if "results" in src or "similar_results" in src:
                # Nested subquery bucket from WebSearchRetriever.
                subquery = (src.get("subquery") or "").strip()
                subquery = subquery or "(unlabeled subquery)"
                srcs: List[Dict[str, Any]] = []
                for key in ("results", "similar_results"):
                    for item in src.get(key) or []:
                        if isinstance(item, dict):
                            srcs.append(item)
                buckets.append((subquery, srcs))
            else:
                flat.append(src)
        if flat and not buckets:
            buckets.append(("(retrieved sources)", flat))
        return buckets

    @staticmethod
    def _parse_subqueries(plan_result: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(plan_result)
        except (json.JSONDecodeError, TypeError):
            return []
        if isinstance(data, dict) and isinstance(data.get("subqueries"), list):
            return [s for s in data["subqueries"] if isinstance(s, dict)]
        return []

    @staticmethod
    def _source_text(src: Dict[str, Any]) -> str:
        parts = [src.get("title") or "", src.get("text") or ""]
        highlights = src.get("highlights")
        if isinstance(highlights, list):
            parts.append(" ".join(str(h) for h in highlights))
        return " ".join(parts)

    def _tokenize(self, text: str) -> Set[str]:
        return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS

    def _numbers(self, text: str) -> Set[str]:
        return set(_NUMBER_RE.findall(text.lower()))

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join((query or "").lower().split())
