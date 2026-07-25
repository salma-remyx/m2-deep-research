"""Evidence stressor -- a controlled environment for poor-quality evidence.

Adapted from **DeepStress: Stress-Testing Deep Search Agents**
(arXiv:2607.13920v1). DeepStress replaces a search agent's retrieval module
with a *controlled synthetic environment* that lets you dial the **frequency**
of challenging evidence along the three dimensions that affect document
reliability -- *trustworthiness*, *relevance*, and *factuality* -- and then
measures how robustly the agent handles it.

This is a **Mode 2 (adapted port)** of that mechanism for this deep-research
pipeline. What is preserved and what is substituted:

* **Preserved (the core mechanism).** A controlled *rate* of challenging
  evidence injected along the *same three dimensions* DeepStress controls,
  used to probe an existing evidence-handling component -- here the
  post-synthesis :class:`~src.agents.auditor.ReportAuditor` -- rather than a
  live search agent. The probe asks the question DeepStress asks: when
  untrustworthy / irrelevant / conflicting documents enter the evidence set,
  does the downstream component catch them?

* **Substituted out (auxiliary machinery that does not fit a small PR).**
  - DeepStress *replaces the whole retrieval module* and runs full QA
    benchmarks (HotpotQA, BrowseCompPlus) across several search agents. That
    controlled-retrieval swap and the benchmark harness are cut: instead this
    port *augments* a captured source set
    (:attr:`~src.agents.supervisor.SupervisorAgent._gathered_sources`) with a
    controlled fraction of synthetic challenging sources. Evaluation over
    multiple agents / a separate benchmark suite is downstream-PR territory.
  - DeepStress generates challenging documents to spec per query with an LLM;
    here they are produced by a **parameter-free, deterministic generator**
    (templates + a seeded RNG keyed off the clean sources), so no extra API
    key is needed and the behavior is fully reproducible offline.

The honest limitation this surfaces is itself a DeepStress-style finding: the
auditor is a *lexical* grounding proxy, so it catches fabricated URLs
(trustworthiness) and off-topic claims (relevance) but **cannot** detect a
factuality conflict -- a claim that reuses a clean source's words with one
number flipped reads as "grounded." DeepStress is explicitly about "the
interactions between conflicting parametric and retrieved knowledge"; this
probe documents exactly that blind spot.
"""

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# The three document-reliability dimensions DeepStress controls.
TRUSTWORTHINESS = "trustworthiness"  # fabricated: a URL the retriever never returned
RELEVANCE = "relevance"  # off-topic: text unrelated to the research query
FACTUALITY = "factuality"  # conflicting: a numeric fact that contradicts a clean source
DIMENSIONS: Tuple[str, ...] = (TRUSTWORTHINESS, RELEVANCE, FACTUALITY)

# A deliberately off-domain passage used to build relevance challenges. It is
# unrelated to any plausible deep-research query so its terms never overlap a
# real source corpus -- which is exactly what makes it detectable.
_OFFTOPIC_TEXT = (
    "A detailed guide to sourdough bread: maintaining 75 percent hydration "
    "and a 24 hour cold fermentation improves the open crumb structure."
)
_OFFTOPIC_CLAIM = (
    "Sourdough hydration of 75 percent over 24 hours improves the crumb."
)

_NUMBER_RE = re.compile(r"\d+")


@dataclass
class InjectedChallenge:
    """One challenging source injected into the evidence set."""

    dimension: str
    source: Dict[str, Any]
    # For relevance/factuality, a single-sentence claim that leans on this
    # source (None for trustworthiness, which is probed via a citation).
    claim: Optional[str] = None


@dataclass
class ChallengeOutcome:
    """Whether the auditor caught one injected challenge."""

    dimension: str
    detected: bool
    detail: str


@dataclass
class StressReport:
    """Outcome of a stress probe over an auditor + clean source set."""

    n_clean: int
    outcomes: List[ChallengeOutcome] = field(default_factory=list)
    injected_sources: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_injected(self) -> int:
        return len(self.outcomes)

    @property
    def challenge_rate_actual(self) -> float:
        total = self.n_clean + self.n_injected
        return self.n_injected / total if total else 0.0

    @property
    def detection_rate(self) -> float:
        """Fraction of injected challenges the auditor flagged."""
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.detected) / len(self.outcomes)

    def detection_by_dimension(self) -> Dict[str, float]:
        """Per-dimension detection fraction (1.0 = always caught)."""
        grouped: Dict[str, List[bool]] = {}
        for o in self.outcomes:
            grouped.setdefault(o.dimension, []).append(o.detected)
        return {dim: sum(v) / len(v) for dim, v in grouped.items()}


class EvidenceStressor:
    """Inject controlled challenging evidence to stress-test evidence handling.

    Parameter-free and deterministic: given a clean source set and a research
    query, it produces challenging sources along DeepStress's three dimensions
    at a controlled ``challenge_rate``. Pair with
    :class:`~src.agents.auditor.ReportAuditor` via :meth:`probe` to measure how
    many injected challenges the grounding auditor catches.
    """

    def inject(
        self,
        clean_sources: List[Dict[str, Any]],
        research_query: str,
        challenge_rate: float = 0.5,
        seed: int = 0,
    ) -> Tuple[List[Dict[str, Any]], List[InjectedChallenge]]:
        """Return ``(stressed_sources, challenges)``.

        ``stressed_sources`` is the clean set plus one synthetic challenging
        source per injected challenge -- the controlled environment that
        stands in for DeepStress's replaced retrieval module. ``challenge_rate``
        is the target fraction of the clean set to challenge (at least one when
        positive).
        """
        clean = [s for s in clean_sources if isinstance(s, dict)]
        challenges = self._build_challenges(clean, research_query, challenge_rate, seed)
        stressed = clean + [c.source for c in challenges]
        return stressed, challenges

    def probe(
        self,
        auditor: Any,
        clean_sources: List[Dict[str, Any]],
        research_query: str,
        challenge_rate: float = 0.5,
        seed: int = 0,
    ) -> StressReport:
        """Inject challenging evidence and measure how much ``auditor`` catches.

        For each injected challenge we build a minimal report that *leans on*
        it (a citation for a fabricated URL; a claim for off-topic / conflicting
        text) and audit that report against the **clean** sources -- the real
        evidence. A robust auditor flags the claim/citation; a lexical proxy
        grounds factuality conflicts and lets them through.
        """
        clean = [s for s in clean_sources if isinstance(s, dict)]
        challenges = self._build_challenges(clean, research_query, challenge_rate, seed)
        outcomes: List[ChallengeOutcome] = []
        for ch in challenges:
            report = self._challenge_report(ch)
            result = auditor.audit(report, clean)
            detected, detail = self._detect(ch, result)
            outcomes.append(ChallengeOutcome(ch.dimension, detected, detail))
        return StressReport(
            n_clean=len(clean),
            outcomes=outcomes,
            injected_sources=[c.source for c in challenges],
        )

    # -- challenge generation ---------------------------------------------

    def _build_challenges(
        self,
        clean: List[Dict[str, Any]],
        research_query: str,
        challenge_rate: float,
        seed: int,
    ) -> List[InjectedChallenge]:
        if not clean or challenge_rate <= 0:
            return []
        n_inject = max(1, int(round(challenge_rate * len(clean))))
        # Cap so a huge rate cannot dwarf the clean set unrealistically.
        n_inject = min(n_inject, len(DIMENSIONS) * 4)
        rng = random.Random(seed)
        dims = [DIMENSIONS[i % len(DIMENSIONS)] for i in range(n_inject)]
        return [
            self._make_challenge(dim, clean, research_query, i, rng)
            for i, dim in enumerate(dims)
        ]

    def _make_challenge(
        self,
        dimension: str,
        clean: List[Dict[str, Any]],
        research_query: str,
        i: int,
        rng: random.Random,
    ) -> InjectedChallenge:
        if dimension == TRUSTWORTHINESS:
            url = f"https://fabricated-stress.invalid/unverified-{i}"
            source: Dict[str, Any] = {
                "url": url,
                "title": f"Unverified Source {i}",
                "text": "Figures and claims presented without verifiable corroboration.",
                "highlights": ["unverified claim without corroboration"],
            }
            return InjectedChallenge(dimension=dimension, source=source, claim=None)

        if dimension == RELEVANCE:
            url = f"https://offtopic-stress.example/irrelevant-{i}"
            source = {
                "url": url,
                "title": f"Unrelated Topic {i}",
                "text": _OFFTOPIC_TEXT,
                "highlights": ["sourdough hydration fermentation"],
            }
            return InjectedChallenge(
                dimension=dimension, source=source, claim=_OFFTOPIC_CLAIM
            )

        # FACTUALITY: reuse a clean source's wording with one number flipped so
        # the claim is lexically "grounded" in the clean corpus -- the conflict
        # a lexical auditor cannot see.
        anchor = clean[i % len(clean)]
        base_text = str(anchor.get("text") or "")
        conflict_text = self._flip_first_number(base_text, rng)
        url = f"https://conflicting-stress.example/figures-{i}"
        source = {
            "url": url,
            "title": f"{anchor.get('title', 'Source')} (conflicting figures)",
            "text": conflict_text,
            "highlights": anchor.get("highlights") or [],
        }
        return InjectedChallenge(dimension=dimension, source=source, claim=conflict_text)

    @staticmethod
    def _flip_first_number(text: str, rng: random.Random) -> str:
        """Return ``text`` with one numeric token changed to a different value."""
        numbers = _NUMBER_RE.findall(text)
        if not numbers:
            return text.rstrip(".") + ". Actually the reported figure is 2."
        target = rng.choice(numbers)
        value = int(target)
        wrong = value + 1000 if value < 1000 else max(1, value // 10)
        return text.replace(target, str(wrong), 1)

    # -- probing ----------------------------------------------------------

    @staticmethod
    def _challenge_report(ch: InjectedChallenge) -> str:
        """A minimal report that leans on one injected challenge."""
        if ch.dimension == TRUSTWORTHINESS:
            url = ch.source["url"]
            return f"According to an [unverified source]({url}), the figures hold."
        # relevance / factuality: a numeric claim drawn from the challenge.
        return ch.claim or ""

    @staticmethod
    def _detect(ch: InjectedChallenge, result: Any) -> Tuple[bool, str]:
        """Did ``result`` flag the claim/citation anchored on ``ch``?"""
        if ch.dimension == TRUSTWORTHINESS:
            url = ch.source["url"]
            caught = url in result.unsupported_citations
            return caught, "fabricated-citation flag" if caught else "url accepted as grounded"
        # Numeric claim: detected iff the auditor could not ground it.
        caught = bool(result.unsupported_claims)
        return caught, "unsupported-claim flag" if caught else "claim lexically grounded"
