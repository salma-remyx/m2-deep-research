"""Pre-pipeline research-query refinement via an intent elicitation graph.

Adapted from **G-STEER** -- *Personalized Deep Research Query Refinement with
Graph-Scaffolded Evidence Grounding* (arXiv:2608.05876v1). G-STEER refines a
raw user request into a *personalized research specification* **before** it is
handed to an otherwise-unchanged deep-research agent. It organizes the
"framing factors" a request could be sharpened on (goals, scope, audience,
constraints, preferences, evaluation criteria, ...) as elicitation targets in
an **Intent Elicitation Graph** that captures their dependencies, then learns a
**clarification policy** that decides -- per factor -- whether the available
context already supports it, or whether to retrieve memory, ask the user, or
stop and fold what is known into the refined query.

This is a **Mode 2 (adapted port)** of that mechanism for this deep-research
pipeline:

* G-STEER's **learned clarification policy** (trained on graph-scaffolded
  trajectories spanning diverse factor dependencies and evidence conditions) is
  replaced by a **parameter-free, graph-driven self-elicitation policy**. This
  pipeline is autonomous -- it has no user-in-the-loop and no user memory to
  retrieve -- so G-STEER's three-way decision (retrieve / ask / stop) collapses
  to, per factor: is the factor already *grounded* in the request, or does it
  need a clarifying question that is then *self-answered* from a sensible
  default to form the spec? Self-elicitation is the natural target-native form
  of G-STEER's clarification step when no human is available to answer.
* G-STEER's **learned coverage / evidence-acquisition estimator** is replaced
  by a **parameter-free coverage proxy**: the fraction of graph factors the
  request already grounds (keyword/category overlap), with the un-grounded
  factors surfaced as the clarifying questions G-STEER would ask. This mirrors
  the paper's efficiency result (it asks roughly a third as many questions as a
  strong baseline) -- only the genuinely under-specified factors generate
  questions.
* G-STEER's **training procedure** and its **downstream-DRA benchmark / report
  personalization evaluation** are deliberately **not** ported: the policy here
  is parameter-free (no training), and per the repo's convention evaluation
  belongs in a downstream PR.

The **core mechanism is preserved**: framing factors are organized as a
dependency graph (an :class:`IntentElicitationGraph`), the policy walks them in
dependency order classifying each factor's evidence state, and the output is a
structured, personalized research specification consumed verbatim by the
unchanged downstream pipeline (here it flows into the supervisor, which hands
it to the planning agent).

This module is the pre-pipeline counterpart to
:class:`~src.agents.auditor.ReportAuditor`: the auditor checks a finished
report, the refiner sharpens the request before research begins. It is fully
deterministic and makes no API calls, so it runs on every request and is
unit-testable offline.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


# Tokens of >= 2 lowercase alphanumeric chars. Kept short (>=2, not >=3) so
# question/contrast words like "vs" or "how" survive to drive factor grounding.
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


@dataclass(frozen=True)
class FramingFactor:
    """A single node in the Intent Elicitation Graph.

    Attributes:
        id: Stable factor identifier.
        label: Human-readable factor name used in the rendered spec.
        question: The clarifying question G-STEER's policy would ask to pin
            this factor down.
        keywords: Request terms that, if present, mean the factor is already
            grounded and need not be elicited.
        default: Provisional self-elicited resolution used when the factor is
            not grounded (there is no user to ask).
        depends_on: Other factor ids this factor's meaning depends on; the
            policy resolves dependencies before the dependent.
    """

    id: str
    label: str
    question: str
    keywords: Tuple[str, ...]
    default: str
    depends_on: Tuple[str, ...] = ()


# The default Intent Elicitation Graph: the framing factors G-STEER elicits for
# a deep-research request, with the dependency edges between them. "objective"
# is the request itself, so a non-empty request always grounds it; the policy
# surfaces the remaining factors as the ones worth clarifying.
DEFAULT_FACTORS: Tuple[FramingFactor, ...] = (
    FramingFactor(
        "objective",
        "Objective",
        "What core question or decision must the research inform?",
        ("what", "how", "why", "compare", "evaluate", "analyze", "assess",
         "overview", "impact", "effect", "explain", "review", "guide",
         "investigate", "explore", "study"),
        "Comprehensive analysis of exactly the topic the request names.",
        (),
    ),
    FramingFactor(
        "scope",
        "Scope & boundaries",
        "Which subtopics are in scope, and what should be excluded?",
        ("scope", "focus", "include", "exclude", "specifically", "between",
         "versus", "vs", "compare", "comparison", "across", "subset"),
        "All major subtopics of the topic; none singled out to exclude.",
        ("objective",),
    ),
    FramingFactor(
        "audience",
        "Audience",
        "Who is the primary audience (executive, engineer, general reader)?",
        ("executive", "beginner", "intro", "technical", "engineer",
         "developer", "student", "business", "academic", "researcher",
         "scientist", "manager", "expert", "novice", "layperson"),
        "A general professional audience.",
        ("objective",),
    ),
    FramingFactor(
        "depth",
        "Depth & detail",
        "How detailed should the report be (brief overview vs. deep dive)?",
        ("brief", "short", "deep", "detailed", "comprehensive", "overview",
         "summary", "quick", "thorough", "exhaustive", "primer",
         "introduction", "depth"),
        "Comprehensive depth: both breadth of coverage and technical detail.",
        ("scope",),
    ),
    FramingFactor(
        "time_horizon",
        "Time horizon",
        "Is the focus historical, the current state, or future outlook?",
        ("history", "historical", "forecast", "future", "outlook",
         "prediction", "trends", "roadmap", "timeline", "evolution",
         "recent", "latest", "current", "emerging", "2023", "2024", "2025",
         "2026"),
        "Current state plus relevant history and a near-term outlook.",
        (),
    ),
    FramingFactor(
        "constraints",
        "Constraints",
        "Any format, length, source-type, or methodology constraints?",
        ("citation", "citations", "format", "length", "pages", "words",
         "sources", "peer", "academic", "pdf", "quantitative", "qualitative",
         "methodology", "statistics", "stats", "data", "table", "chart"),
        "An inline-cited markdown report; no constraints beyond that.",
        ("objective",),
    ),
    FramingFactor(
        "success_criteria",
        "Success criteria",
        "What would make this report complete and actionable for the reader?",
        ("decision", "recommendation", "actionable", "deliverable",
         "criteria", "choose", "invest", "select", "justify",
         "stakeholder", "metrics", "goal", "objective"),
        "A thorough, well-cited report the reader can act on.",
        ("objective", "audience"),
    ),
)


@dataclass(frozen=True)
class IntentElicitationGraph:
    """The Intent Elicitation Graph: framing factors plus their dependencies."""

    factors: Tuple[FramingFactor, ...] = DEFAULT_FACTORS

    def in_dependency_order(self) -> List[FramingFactor]:
        """Return factors with each dependency appearing before its dependents.

        Topological sort over :pyattr:`depends_on`; declared order is the
        tie-breaker. Cycles (which the defaults never contain) are tolerated --
        a node is emitted the first time it is reached.
        """
        by_id: Dict[str, FramingFactor] = {f.id: f for f in self.factors}
        ordered: List[FramingFactor] = []
        done: Set[str] = set()

        def visit(factor: FramingFactor) -> None:
            if factor.id in done:
                return
            for dep in factor.depends_on:
                dep_factor = by_id.get(dep)
                if dep_factor is not None:
                    visit(dep_factor)
            done.add(factor.id)
            ordered.append(factor)

        for factor in self.factors:
            visit(factor)
        return ordered


@dataclass
class FactorState:
    """A framing factor plus the policy's classification of it for a request."""

    factor: FramingFactor
    grounded: bool
    resolution: str


@dataclass
class RefinementResult:
    """Outcome of refining a single request through the elicitation graph."""

    original_query: str
    factors: List[FactorState] = field(default_factory=list)
    refined_spec: str = ""

    @property
    def grounded_count(self) -> int:
        return sum(1 for f in self.factors if f.grounded)

    @property
    def elicited_count(self) -> int:
        return sum(1 for f in self.factors if not f.grounded)

    @property
    def coverage(self) -> float:
        """Fraction of framing factors the request already grounds."""
        total = len(self.factors)
        return (self.grounded_count / total) if total else 1.0


class QueryRefinementAgent:
    """Refine a raw research request into an intent-grounded specification.

    Walks the :class:`IntentElicitationGraph` in dependency order. For each
    framing factor the parameter-free policy asks one question -- is this factor
    already grounded in the request? -- and either keeps the grounded signal or
    provisions a self-elicited default (there is no user to ask). The result is
    a structured research specification the unchanged downstream pipeline
    consumes.
    """

    def __init__(self, factors: Optional[Sequence[FramingFactor]] = None):
        self.graph = IntentElicitationGraph(
            factors=tuple(factors) if factors is not None else DEFAULT_FACTORS
        )

    def refine(self, query: str, context: str = "") -> str:
        """Refine ``query`` into a personalized research specification string.

        Args:
            query: The raw user research request.
            context: Optional extra context (e.g. CLI-provided preferences)
                that can ground additional factors, standing in for the user
                memory G-STEER would otherwise retrieve.

        Returns:
            A markdown research specification beginning with the operative
            research goal, followed by the per-factor framing.
        """
        return self.analyze(query, context).refined_spec

    def analyze(self, query: str, context: str = "") -> RefinementResult:
        """Classify each framing factor for ``query`` and render the refined spec."""
        tokens = self._tokenize(f"{query} {context}")
        factors = self.graph.in_dependency_order()
        states: List[FactorState] = []
        for factor in factors:
            matched = [kw for kw in factor.keywords if self._keyword_matches(kw, tokens)]
            # The objective is whatever the request asks, so a non-empty
            # request always grounds it -- only the other factors can be
            # under-specified.
            grounded = bool(matched) or (factor.id == "objective" and bool(query.strip()))
            if grounded:
                shown = ", ".join(matched) if matched else "the request itself"
                resolution = f"Grounded in the request (matched: {shown})."
            else:
                resolution = factor.default
            states.append(FactorState(factor=factor, grounded=grounded, resolution=resolution))
        result = RefinementResult(original_query=query, factors=states)
        result.refined_spec = self.format_spec(result)
        return result

    def format_spec(self, result: RefinementResult) -> str:
        """Render a refinement result as a markdown research specification."""
        lines: List[str] = []
        # Lead with the operative research goal so the downstream supervisor /
        # planning agent reads a clear directive first.
        goal = result.original_query.strip() or "(no request provided)"
        lines.append(goal)
        lines.extend(("", "---", "", "## Research framing", ""))
        lines.append(
            "> Pre-pipeline intent refinement. The request was expanded through"
        )
        lines.append(
            "> an intent-elicitation graph of framing factors; factors the"
        )
        lines.append(
            "> request already grounds are kept, the rest are provisionally"
        )
        lines.append(
            "> clarified (self-elicited -- this pipeline has no user-in-the-loop)."
        )
        lines.append(
            "> Adapted from G-STEER (arXiv:2608.05876v1), Mode 2 adapted port."
        )
        lines.append("")
        total = len(result.factors)
        lines.append(
            f"**Intent coverage:** {result.grounded_count}/{total} factors "
            f"grounded in the request; {result.elicited_count} provisionally "
            f"clarified."
        )
        lines.append("")

        grounded = [f for f in result.factors if f.grounded]
        elicited = [f for f in result.factors if not f.grounded]
        if grounded:
            lines.append("**Already grounded in the request:**")
            for f in grounded:
                lines.append(f"- **{f.factor.label}:** {f.resolution}")
            lines.append("")
        if elicited:
            lines.append("**Provisionally clarified (override if needed):**")
            for f in elicited:
                lines.append(
                    f"- **{f.factor.label}:** _Q: {f.factor.question}_ "
                    f"Provisional: {f.resolution}"
                )
            lines.append("")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _keyword_matches(self, keyword: str, tokens: Set[str]) -> bool:
        """Whether a framing-factor keyword is present in the request tokens.

        Exact token match, plus a light plural/derivative tolerance for longer
        keywords only (so "engineers" grounds the "engineer" keyword), while
        short ambiguous words such as "vs" or "how" stay exact to avoid slop.
        """
        if keyword in tokens:
            return True
        return len(keyword) >= 4 and any(tok.startswith(keyword) for tok in tokens)

    def _tokenize(self, text: str) -> Set[str]:
        return set(_TOKEN_RE.findall(text.lower()))
