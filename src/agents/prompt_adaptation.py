"""Cross-component prompt adaptation for the research loop.

Adapted from **GRADRAG: Cross-Component Prompt Adaptation for Coordinated
Multi-Agent RAG** (arXiv:2607.21324v1). GRADRAG models a RAG pipeline as a
computational graph and propagates **structured evaluation feedback to update
upstream agents' prompts**. Its triad is:

* an **Evaluator** that critiques the synthesized answer and its evidence,
* a **Prompt Optimizer** that turns that critique into actionable prompt
  updates for adaptive agents (retrievers, planners, answerers), and
* an **early-stopping** rule that halts refinement once the output is judged
  satisfactory.

This deep-research pipeline already supplies the Evaluator: the post-synthesis
:class:`~src.agents.auditor.ReportAuditor` produces an
:class:`~src.agents.auditor.AuditResult` (grounded vs. total citations,
unsupported claims). This module supplies the other two pieces -- a
parameter-free **Prompt Optimizer** (:class:`PromptAdapter`) that converts an
``AuditResult`` into upstream prompt directives, plus the early-stopping
decision -- and :class:`~src.agents.supervisor.SupervisorAgent` wires them into
the ``end_turn`` branch of its research loop so grounding gaps feed back into
the next planning/retrieval pass instead of being reported and ignored.

This is a **Mode 2 (adapted port)** of GRADRAG's mechanism:

* GRADRAG's **LLM Prompt Optimizer** (which rewrites agent prompts from
  learned / LLM-judged feedback) is replaced by a **parameter-free prompt
  adaptation proxy** that derives concrete directives from the auditor's
  unsupported citations and claims. This mirrors how this pipeline already
  replaced BrainPilot's LLM fabrication judge with a parameter-free grounding
  proxy in :mod:`src.agents.auditor`: deterministic, needs no extra API keys,
  is unit-testable offline, and runs on every report.
* GRADRAG's **SQUALITY / QMSUM benchmark suite and LLM pairwise-judge
  evaluation** are deliberately **not** ported -- evaluation belongs in a
  downstream PR; here the auditor's grounding score is the in-loop signal.

The core mechanism is preserved: structured evaluation feedback (the audit)
propagates upstream as concrete prompt adaptation for the adaptive agent
(the planning agent, which drives retrieval), with early stopping when
grounding is clean and a refinement budget that defaults to two iterations
(GRADRAG reports most gains are realized within two refinement iterations).
"""

import re
from dataclasses import dataclass, field
from typing import List

from src.agents.auditor import AuditResult


# GRADRAG reports that most gains are realized within two refinement iterations.
DEFAULT_MAX_REFINEMENTS = 2

# Cap how many focus terms we push into the adapted prompt, keeping it focused.
DEFAULT_MAX_FOCUS_TERMS = 8

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Common words ignored when distilling unsupported claims into focus terms, so
# the adapted prompt carries topic signal rather than filler. (Deliberately a
# small local set -- this module's focus-term view, decoupled from the
# auditor's own stopword list.)
_STOPWORDS = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over
    under about as is are was were be been being this that these those it its
    their our your his her they them we you he she him not no nor so than too
    very can could should would may might must will shall do does did has have
    had more most less least many much few several also however which who whom
    whose what when where why how during while across among between within
    without via per using used use new one two three first second next last
    according based recent currently reportedly said says estimated projected
    expected predicted reached nearly almost around about according
    """.split()
)


@dataclass
class PromptDirective:
    """A GRADRAG prompt-adaptation directive produced from an audit.

    Carries the Prompt Optimizer's output (the feedback to propagate upstream
    to the planning agent and supervisor) plus the Evaluator's early-stopping
    decision.
    """

    should_stop: bool
    reason: str
    feedback: str = ""
    focus_terms: List[str] = field(default_factory=list)
    refinement: int = 0

    @property
    def needs_refinement(self) -> bool:
        """True when the loop should re-enter with adapted prompts."""
        return not self.should_stop


class PromptAdapter:
    """Parameter-free GRADRAG Prompt Optimizer + early-stopping rule.

    Converts an :class:`~src.agents.auditor.AuditResult` (the Evaluator's
    structured critique) into an upstream prompt directive for the pipeline's
    adaptive agent, and decides whether to early-stop or refine once more.
    """

    def __init__(
        self,
        max_refinements: int = DEFAULT_MAX_REFINEMENTS,
        max_focus_terms: int = DEFAULT_MAX_FOCUS_TERMS,
    ):
        self.max_refinements = max_refinements
        self.max_focus_terms = max_focus_terms

    def adapt(self, audit_result: AuditResult, refinement: int) -> PromptDirective:
        """Turn an audit into a prompt directive + early-stop decision.

        Args:
            audit_result: The Evaluator's grounding audit of the last report.
            refinement: How many audit-driven refinements have already run.

        Returns:
            A :class:`PromptDirective`. ``should_stop`` is True when grounding
            is clean (satisfactory -> early stop), when there is nothing to
            adapt against (unverifiable audit), or when the refinement budget
            is exhausted. Otherwise it carries feedback + focus terms and the
            loop refines once more.
        """
        # Early stop: output judged satisfactory.
        if audit_result.passed:
            return PromptDirective(
                should_stop=True,
                reason="grounding audit passed - report is satisfactory",
                refinement=refinement,
            )
        # Nothing to adapt against: no sources were captured to verify.
        if not audit_result.verifiable:
            return PromptDirective(
                should_stop=True,
                reason="audit unverifiable (no sources) - nothing to adapt against",
                refinement=refinement,
            )
        # Budget exhausted: ship the best-effort report rather than loop forever.
        if refinement >= self.max_refinements:
            return PromptDirective(
                should_stop=True,
                reason=(
                    f"refinement budget exhausted ({refinement}/"
                    f"{self.max_refinements}) - shipping best-effort report"
                ),
                refinement=refinement,
            )

        # Build the cross-component adaptation from the gaps the audit found.
        focus = self._focus_terms(audit_result)
        feedback = self._build_feedback(audit_result, focus, refinement)
        return PromptDirective(
            should_stop=False,
            reason=(
                f"{len(audit_result.unsupported_citations)} ungrounded citation(s), "
                f"{len(audit_result.unsupported_claims)} unsupported claim(s)"
            ),
            feedback=feedback,
            focus_terms=focus,
            refinement=refinement,
        )

    def planning_advisory(self, directive: PromptDirective) -> str:
        """Prompt text to append to the adaptive planning agent's system prompt.

        Empty when there is nothing focused to add, so callers can append
        unconditionally.
        """
        if not directive.focus_terms:
            return ""
        bullets = "\n".join(f"- {term}" for term in directive.focus_terms)
        return (
            "\n\n## Adaptive focus (from grounding audit)\n"
            "The previous synthesis had grounding gaps. Generate at least one "
            "Exa-optimized subquery targeting each of these topics so the next "
            "retrieval covers them with primary sources:\n" + bullets
        )

    # -- internals ---------------------------------------------------------

    def _focus_terms(self, audit_result: AuditResult) -> List[str]:
        """Distill the unsupported claims into focused re-query topics."""
        bag = set()
        for claim in audit_result.unsupported_claims:
            bag |= self._tokenize(claim)
        terms = sorted(term for term in bag if term not in _STOPWORDS)
        return terms[: self.max_focus_terms]

    def _build_feedback(
        self, audit_result: AuditResult, focus: List[str], refinement: int
    ) -> str:
        """Render the cross-component feedback injected into the next turn."""
        lines = [
            f"Grounding audit found gaps after synthesis "
            f"(refinement {refinement + 1} of {self.max_refinements}).",
            f"- {len(audit_result.unsupported_citations)} of "
            f"{audit_result.total_citations} cited URLs were not among the "
            f"retrieved sources (possible fabrications).",
            f"- {len(audit_result.unsupported_claims)} numeric claim(s) had no "
            f"lexical support in the retrieved evidence.",
        ]
        if focus:
            lines.append(
                "Re-plan the next search to close these gaps; generate "
                "subqueries targeting:"
            )
            for term in focus:
                lines.append(f"  - {term}")
        else:
            lines.append(
                "Re-plan the next search to retrieve primary sources that back "
                "the report's citations and numeric claims."
            )
        lines.append(
            "Prefer authoritative primary sources; replace any citation that no "
            "retrieved source supports."
        )
        return "\n".join(lines)

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(_TOKEN_RE.findall(text.lower()))
