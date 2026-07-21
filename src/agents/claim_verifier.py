"""Post-audit claim re-verification via iterative query rewriting.

Adapted from the **iterative query rewriting** retrieval mechanism in
*AgentKGV: Agentic LLM-RAG Framework with Two-Stage Training for the Fact
Verification of Knowledge Graphs* (arXiv:2607.09092v1). AgentKGV closes
fact-verification gaps by rewriting a verification query across surface forms
and re-retrieving until a supporting document is found -- directly attacking
the *surface-form mismatch* that defeats single-turn retrieval.

This is a **Mode 2 (adapted port)** of that retrieval mechanism for this
deep-research pipeline. The grounding auditor
(:class:`src.agents.auditor.ReportAuditor`) already *detects* claims it cannot
trace to retrieved sources, but it only appends a warning -- it never tries to
*recover* them. This module feeds the auditor's ``unsupported_claims`` back
through AgentKGV's iterative rewrite-and-re-retrieve loop over the pipeline's
existing Exa retrieval, so a claim that missed only because of phrasing gets a
second, third, ... chance to ground.

**What is ported (the core mechanism).** An iterative loop that, for each
unsupported claim, emits successive surface-form rewrites of a verification
query and re-retrieves over Exa, stopping as soon as a retrieved source grounds
the claim (under the same lexical-overlap criterion the auditor uses) or the
rewrite budget is spent.

**What is substituted (the paper's auxiliaries -- cut per Mode 2).**

* AgentKGV's **learned query rewriter** -- a small model trained via turn-level
  SFT distillation from a large teacher plus trajectory-level GRPO that
  optimizes the search policy -- is replaced by a **parameter-free rewriter**:
  deterministic surface-form transformations (stopword-stripped key-term
  extraction, quoted longest content phrase, numeric/entity anchors). This is
  the same substitution shape the BrainPilot-auditor port used for its LLM
  judge: no extra API keys, fully deterministic, unit-testable offline.
* AgentKGV's **dynamic routing** across multiple retrieval backends collapses
  to the single Exa path this repo already exposes
  (:class:`src.tools.exa_tool.ExaTool`).
* AgentKGV's **GRPO search-policy optimizer** and **SFT distillation trainer**
  are cut entirely -- the repo hosts no training loop. The iterative *loop*
  (the object GRPO was optimizing) is what is ported; a fixed, bounded rewrite
  order stands in for the learned policy.
* AgentKGV's **T-REx long-tail benchmark evaluation** is out of scope
  (evaluation belongs in a downstream PR); tests here assert the wiring and the
  recover/miss behavior on deterministic fixtures, not benchmark numbers.

The pass is best-effort and bounded (at most ``max_claims`` claims, each tried
against at most ``max_rewrites`` query variants returning ``num_results``
sources), so it never blocks report delivery and its cost is predictable.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.tools.exa_tool import ExaTool

# Tokens of >= 3 lowercase alphanumeric chars (matches the auditor's tokenizer).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Numeric runs -- numbers survive surface-form rewording, so they anchor
# retrieval on exactly the facts where mismatch is worst.
_NUMBER_RE = re.compile(r"\d[\d,./]*")

# Capitalized words -- proper nouns / entities are strong, stable query terms.
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

# Common words ignored when judging whether a source grounds a claim, so trivial
# keyword overlap does not read as support. Mirrors the auditor's stopword bar
# so a "recovered" claim is one the auditor itself would have accepted had it
# seen the re-retrieved source.
_STOPWORDS = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over
    under about as is are was were be been being this that these those it its
    their our your his her they them we you he she him not no nor so than too
    very can could should would may might must will shall do does did has have
    had more most less least many much few several also however which who whom
    whose what when where why how during while across among between within
    without via per using used use new one two three first second next according
    based recent currently reportedly said says estimated projected expected
    predicted reached million billion
    """.split()
)


@dataclass
class ClaimVerification:
    """Outcome of re-verifying a single unsupported claim."""

    claim: str
    verified: bool
    query_used: Optional[str]
    supporting_url: Optional[str]
    rewrites_attempted: int


@dataclass
class VerificationReport:
    """Aggregate outcome of re-verifying a batch of unsupported claims."""

    results: List[ClaimVerification] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def verified_count(self) -> int:
        return sum(1 for r in self.results if r.verified)


class ClaimVerifier:
    """Recover unsupported claims by iterative query rewriting over Exa.

    Given the ``unsupported_claims`` produced by the grounding auditor, rewrite
    each claim into successive verification queries, re-retrieve over Exa, and
    report whether any retrieved source grounds the claim. This is AgentKGV's
    iterative query-rewriting loop with the learned rewriter swapped for a
    parameter-free one.
    """

    def __init__(
        self,
        exa: Optional[ExaTool] = None,
        min_token_overlap: int = 3,
        max_rewrites: int = 3,
        max_claims: int = 5,
        num_results: int = 5,
    ):
        # Injected so tests can drive the loop offline with a fake Exa.
        self.exa = exa if exa is not None else ExaTool()
        self.min_token_overlap = min_token_overlap
        self.max_rewrites = max_rewrites
        self.max_claims = max_claims
        self.num_results = num_results

    def verify(self, claims: Sequence[str]) -> VerificationReport:
        """Re-verify up to ``max_claims`` unsupported claims."""
        results = [self._verify_one(claim) for claim in list(claims)[: self.max_claims]]
        return VerificationReport(results=results)

    def format_section(self, report: VerificationReport) -> str:
        """Render a verification report as a markdown section to append."""
        lines: List[str] = [
            "",
            "### Unsupported-claim re-verification (iterative query rewriting)",
            "",
            "> Best-effort pass that re-checks each unsupported claim by rewriting",
            "> its verification query across surface forms and re-retrieving over",
            "> Exa until a retrieved source grounds it. Adapted from AgentKGV's",
            "> iterative query rewriting (arXiv:2607.09092v1).",
            "",
        ]
        if not report.results:
            lines.append("- _No unsupported claims were re-verified._")
            return "\n".join(lines)

        for r in report.results:
            snippet = r.claim.strip()[:140]
            if r.verified:
                url = r.supporting_url or "a retrieved source"
                lines.append(
                    f'- ✅ Recovered "{snippet}" — grounded after '
                    f"{r.rewrites_attempted} rewrite(s) via `{url}`."
                )
            else:
                lines.append(
                    f'- ❌ Still unsupported "{snippet}" — '
                    f"{r.rewrites_attempted} rewrite(s) found no grounding."
                )
        lines.append("")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _verify_one(self, claim: str) -> ClaimVerification:
        """Iterate query rewrites for one claim until a source grounds it."""
        queries = self._claim_queries(claim)
        for attempt, query in enumerate(queries, start=1):
            for source in self._search(query):
                if self._claim_grounded_by(claim, source):
                    return ClaimVerification(
                        claim=claim,
                        verified=True,
                        query_used=query,
                        supporting_url=source.get("url"),
                        rewrites_attempted=attempt,
                    )
        return ClaimVerification(
            claim=claim,
            verified=False,
            query_used=None,
            supporting_url=None,
            rewrites_attempted=len(queries),
        )

    def _claim_queries(self, claim: str) -> List[str]:
        """Ordered surface-form rewrites of a verification query.

        Deterministic stand-ins for AgentKGV's learned rewriter. The first
        variant is the focused key-term form; if it misses (surface-form
        mismatch), later variants re-anchor on exact phrases, numbers, and
        entities that are more stable across paraphrase.
        """
        tokens = [t for t in _TOKEN_RE.findall(claim.lower()) if t not in _STOPWORDS]
        numbers = _NUMBER_RE.findall(claim)
        proper = _PROPER_RE.findall(claim)

        queries: List[str] = []
        if tokens:
            # 1. Stopword-stripped key-term surface form.
            queries.append(" ".join(tokens))
            # 2. Quoted exact-phrase on the longest contiguous content run --
            #    forces a phrase match that defeats term-scatter mismatch.
            phrase = self._longest_content_phrase(claim)
            if phrase:
                queries.append(f'"{phrase}"')
        if numbers:
            # 3. Numeric anchor -- numbers rarely paraphrase, so they pin the
            #    specific fact being verified.
            queries.append(" ".join(numbers))
        if proper:
            # 4. Entity anchor -- proper nouns disambiguate the subject.
            queries.append(" ".join(proper))

        # De-duplicate while preserving the rewrite order, then cap to budget.
        seen: set = set()
        ordered: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                ordered.append(q)
        return ordered[: self.max_rewrites]

    def _longest_content_phrase(self, claim: str) -> str:
        """Longest contiguous run of content (non-stopword, >=3 char) words."""
        words = re.findall(r"[A-Za-z0-9]+", claim)
        best: List[str] = []
        current: List[str] = []
        for word in words:
            lowered = word.lower()
            if len(lowered) >= 3 and lowered not in _STOPWORDS:
                current.append(word)
            else:
                if len(current) > len(best):
                    best = list(current)
                current = []
        if len(current) > len(best):
            best = current
        return " ".join(best)

    def _claim_grounded_by(self, claim: str, source: Dict[str, Any]) -> bool:
        """True if ``source`` shares enough key terms with ``claim``.

        Mirrors :meth:`ReportAuditor._claim_is_grounded` so the grounding bar
        for a re-retrieved source matches the bar the auditor applies to the
        originally retrieved set.
        """
        claim_tokens = set(_TOKEN_RE.findall(claim.lower())) - _STOPWORDS
        if not claim_tokens:
            return True
        parts = [source.get("title") or "", source.get("text") or ""]
        highlights = source.get("highlights")
        if isinstance(highlights, list):
            parts.append(" ".join(str(h) for h in highlights))
        source_tokens = set(_TOKEN_RE.findall(" ".join(parts).lower()))
        return len(claim_tokens & source_tokens) >= self.min_token_overlap

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Run one Exa search and return formatted sources (empty on error)."""
        try:
            raw = self.exa.search(query=query, num_results=self.num_results)
            return self.exa.format_results(raw)
        except Exception:
            return []
