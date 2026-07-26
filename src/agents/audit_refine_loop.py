"""Outer self-improvement loop that closes the audit -> refine gap.

Adapted from **AREX: Towards a Recursively Self-Improving Agent for Deep
Research** (arXiv:2607.21461v1). AREX alternates an *inner* research loop that
builds a provisional answer with an *outer self-improvement loop* that audits
the answer constraint-wise, identifies unresolved claims, and launches
targeted follow-up research to resolve them.

This is a **Mode 2 (adapted port)** for this deep-research pipeline:

* AREX's *learned autonomous context-update tool* (a model trained to compress
  growing interaction history into a compact improvement state preserving
  verified evidence and unresolved constraints) is replaced by a
  **parameter-free proxy**: the existing
  :class:`~src.agents.auditor.AuditResult` already *is* a compact improvement
  state -- ``unsupported_claims`` are the unresolved constraints and the
  grounded citations are the verified evidence. No extra model or API key.
* AREX's *learned* targeted-query generator is replaced by a
  **parameter-free derivation**: each unsupported claim's salient terms become
  a focused Exa subquery.
* AREX's learned *reward shaping* ("emphasize key steps where decisive evidence
  is acquired") is replaced by the deterministic trigger "only follow up when
  the audit surfaces genuine gaps, and bound the work".

**Intentionally out of scope.** AREX's agentic mid-training and long-horizon
reinforcement learning, and its dense 4B / 122B-A10B MoE model instantiation,
are training-time machinery this inference-time pipeline does not host. This
port delivers AREX's *inference-time* contribution -- the audit->refine loop
-- as a single bounded follow-up pass wired into the supervisor. Recursion is
capped at one refinement round; raising ``max_rounds`` re-enables the full RSI
recursion at the cost of more API calls per report.

The core mechanism is preserved: an outer loop that consumes a
constraint-wise verification signal (the grounding audit) and dispatches
targeted follow-up research for the gaps it finds, then merges the new
evidence back so the report can be grounded rather than merely flagged.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set


# Same token shape the auditor uses (>= 3 lowercase alphanumeric chars) so the
# claim->subquery derivation shares the auditor's notion of a "term".
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Words ignored when turning a claim into a search query. This is a *superset*
# of the auditor's grounding-overlap stopwords: magnitude words like
# "million"/"billion"/"percent" are useless as search terms (nearly every
# numeric claim carries one), so they are dropped from the *query* even though
# they are kept for *grounding overlap* in the auditor.
_QUERY_STOPWORDS: Set[str] = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over
    under about as is are was were be been being this that these those it its
    their our your his her they them we you he she him not no nor so than too
    very can could should would may might must will shall do does did has have
    had more most less least many much few several also however which who whom
    whose what when where why how during while across among between within
    without via per using used use new one two three first second next according
    based recent currently reportedly said says estimated projected expected
    predicted billion million percent numbers number year years
    """.split()
)


@dataclass
class RefineOutcome:
    """Outcome of one audit -> refine pass over a report."""

    refined: bool = False
    followup_subqueries: List[Dict[str, Any]] = field(default_factory=list)
    new_sources: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: str = ""

    @property
    def ran(self) -> bool:
        """True when targeted follow-up research was actually dispatched."""
        return self.refined and bool(self.followup_subqueries)


class AuditRefineLoop:
    """Close the audit -> refine loop for a research report.

    Given a grounding :class:`~src.agents.auditor.AuditResult` carrying
    unresolved claims, derive focused follow-up subqueries, dispatch them to
    the existing :class:`~src.agents.web_search_retriever.WebSearchRetriever`,
    and surface the new evidence so the report's gaps are *resolved* rather
    than merely flagged by the auditor.
    """

    def __init__(self, max_followups: int = 3, min_query_terms: int = 3):
        self.max_followups = max_followups
        self.min_query_terms = min_query_terms

    def derive_followup_subqueries(
        self,
        unsupported_claims: Sequence[str],
        research_query: str = "",
    ) -> List[Dict[str, Any]]:
        """Turn unsupported claims into targeted Exa subqueries.

        Parameter-free: drops query stopwords, keeps each claim's salient
        terms, and prepends the original research topic for context. Claims
        with too few salient terms (nothing to target) are skipped, and
        near-duplicate queries are de-duplicated.
        """
        subqueries: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        topic = research_query.strip()
        for claim in unsupported_claims:
            terms = self._salient_terms(claim)
            if len(terms) < self.min_query_terms:
                continue
            query = f"{topic} {' '.join(terms)}" if topic else " ".join(terms)
            signature = " ".join(sorted(terms))
            if signature in seen:
                continue
            seen.add(signature)
            subqueries.append(
                {
                    "query": query,
                    "type": "auto",
                    "time_period": "any",
                    "priority": 1,
                }
            )
            if len(subqueries) >= self.max_followups:
                break
        return subqueries

    def run(
        self,
        *,
        audit_result: Any,
        research_query: str,
        retriever: Any,
        sources: Iterable[Any],
    ) -> RefineOutcome:
        """Dispatch targeted follow-up research for an audit's gaps.

        Args:
            audit_result: An :class:`AuditResult`-like object exposing
                ``unsupported_claims``, ``passed`` and ``verifiable``.
            research_query: The original research query, for synthesis context.
            retriever: A :class:`WebSearchRetriever`-like object exposing
                ``retrieve(research_query, subqueries_json)`` and
                ``last_search_results``.
            sources: Sources gathered so far (accepted for interface symmetry;
                the caller merges new evidence back itself).

        Returns:
            A :class:`RefineOutcome`. On any error, or when there are no
            addressable gaps, returns an unrefined outcome so report delivery
            is never blocked.
        """
        del sources  # caller owns the merge; kept for a stable call signature

        outcome = RefineOutcome()
        # A clean audit, or one that could not run, leaves nothing to refine.
        if getattr(audit_result, "passed", True):
            return outcome
        if getattr(audit_result, "verifiable", True) is False:
            # No sources were gathered, so the audit surfaced no specific gaps.
            return outcome

        claims = list(getattr(audit_result, "unsupported_claims", []) or [])
        subqueries = self.derive_followup_subqueries(claims, research_query)
        if not subqueries:
            return outcome

        try:
            synthesis = retriever.retrieve(
                research_query, json.dumps({"subqueries": subqueries})
            )
        except Exception:
            # Follow-up research is best-effort; never block report delivery.
            return outcome

        outcome.followup_subqueries = subqueries
        outcome.synthesis = synthesis or ""
        outcome.new_sources = list(getattr(retriever, "last_search_results", []) or [])
        outcome.refined = True
        return outcome

    def format_outcome(self, outcome: RefineOutcome) -> str:
        """Render a refine outcome as a markdown section for the report."""
        lines: List[str] = ["", "---", "", "## Targeted Follow-up Research", ""]
        lines.append("> Outer self-improvement loop: the grounding audit flagged")
        lines.append("> gaps, so the supervisor launched targeted follow-up")
        lines.append("> searches to resolve them. Adapted from AREX")
        lines.append("> (arXiv:2607.21461v1).")
        lines.append("")
        if not outcome.ran:
            lines.append("_No addressable gaps required follow-up research._")
            return "\n".join(lines)
        lines.append(f"**Follow-up subqueries:** {len(outcome.followup_subqueries)}")
        for sq in outcome.followup_subqueries:
            lines.append(f'- "{sq.get("query", "")}"')
        lines.append("")
        lines.append(f"**New sources gathered:** {len(outcome.new_sources)}")
        for src in outcome.new_sources[:5]:
            url = src.get("url", "")
            title = src.get("title") or url or "source"
            lines.append(f"- [{title}]({url})" if url else f"- {title}")
        if len(outcome.new_sources) > 5:
            lines.append(f"- _…and {len(outcome.new_sources) - 5} more_")
        lines.append("")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _salient_terms(self, claim: str) -> List[str]:
        """Salient, de-duplicated query terms from a claim, in first-seen order."""
        seen: Set[str] = set()
        ordered: List[str] = []
        for token in _TOKEN_RE.findall(claim.lower()):
            if token in _QUERY_STOPWORDS or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered
