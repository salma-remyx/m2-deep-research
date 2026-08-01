"""Budget-to-reward efficiency meter for a research run.

Adapted from the efficiency-MEASUREMENT contribution of *Efficiency Matters in
Autonomous Research* (AREK, arXiv:2607.24647v1). AREK argues that an autonomous-
research system should be judged not only by the quality of its final outcome
but by the *efficiency* of the solution-search process -- how little evaluation
budget a system spends to reach a given reward. It captures this as the area
under the curve (AUC) of the Pareto frontier of budget vs. best-reward-so-far,
and shows that this efficiency is a performance dimension *distinct* from final
outcome quality: a method that eventually reaches the best result may improve
slowly and burn far more budget getting there.

This is a **Mode 2 (adapted port)** of that MEASUREMENT contribution for this
deep-research pipeline:

* AREK's "evaluation budget" (one evaluated candidate solution = one unit) is
  replaced by the supervisor's research-loop **iterations** -- each iteration is
  one Minimax M2.1 call and possibly a web-search tool call, i.e. the real cost
  of producing the report. Tool calls and retrieved sources are tracked as
  supporting cost signals. This is the spec's "iteration = budget axis".
* AREK's task "reward" is replaced by the grounding score from the existing
  :class:`~src.agents.auditor.ReportAuditor` -- the pipeline's outcome-quality
  signal in ``[0, 1]``. This is the spec's "grounding score = reward axis".

The core mechanism is preserved: a parameter-free recorder that pairs each
reward observation with the budget consumed so far, builds the running
best-reward-so-far frontier, and reports its normalized AUC alongside the final
grounding score -- efficiency and outcome quality reported as two distinct
dimensions, exactly as AREK advocates. The repo previously tracked iterations
(in the research loop) and grounding (in the auditor) separately but never
combined them into an efficiency dimension; this module is that combination.

What is deliberately **not** ported (no call site in this repo):

* AREK's adaptive **fluid search** procedure -- a portfolio bandit that
  dynamically allocates a fixed budget across a forest of search processes. That
  needs an edit -> verify -> keep verifier over candidate solutions, which a web
  deep-research pipeline has no analog of. The spec scopes this port to the
  efficiency-MEASUREMENT contribution only.
* AREK's cross-system comparison across twelve tasks. Each system is one
  ``research()`` run here; aggregating many runs into a comparative AUC table is
  a downstream evaluation concern, not a per-report artifact.

Within a single research run the frontier carries the reward observed at the
report's final grounding audit, at the budget consumed to reach it; the AUC
therefore reads as "how little budget this report needed to reach its grounding
quality." The recorder and AUC are general over any number of (budget, reward)
samples, so a harness that audits partial reports at intermediate checkpoints
would produce a denser frontier with no change to this module.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class EfficiencySample:
    """One ``(budget, reward)`` observation on the run's Pareto frontier.

    ``budget`` is the cumulative budget consumed when the reward was observed
    (iterations); ``reward`` is the outcome-quality signal in ``[0, 1]``
    (grounding score).
    """

    budget: float
    reward: float


@dataclass
class EfficiencyReport:
    """Summary of a run's efficiency dimension."""

    final_reward: float
    budget_used: float
    budget_horizon: float
    auc: float
    n_samples: int
    tool_calls: int = 0
    sources: int = 0
    samples: List[EfficiencySample] = field(default_factory=list)

    @property
    def verifiable(self) -> bool:
        """True when at least one reward was observed against a real horizon."""
        return self.n_samples > 0 and self.budget_horizon > 0


class EfficiencyMeter:
    """Record budget -> reward samples and compute AREK's efficiency AUC.

    The meter mirrors the supervisor's existing recorders
    (:class:`~src.agents.research_trace.ResearchTrace`,
    :class:`~src.agents.auditor.ReportAuditor`): it is constructed once in
    :meth:`~src.agents.supervisor.SupervisorAgent.__init__`, reset per run,
    fed one budget tick per research-loop iteration plus a reward when the
    grounding audit produces a score, and rendered into a ``## Search
    Efficiency`` section appended to the delivered report.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self, budget_horizon: float = 10.0) -> None:
        """Clear state for a new run.

        Args:
            budget_horizon: The maximum budget of the run (research-loop
                ``max_iterations``), used to normalize the AUC to ``[0, 1]``.
        """
        self.budget_horizon: float = max(float(budget_horizon), 1e-9)
        self._iterations: int = 0
        self._tool_calls: int = 0
        self._sources: int = 0
        self._samples: List[EfficiencySample] = []

    # -- recording --------------------------------------------------------

    @property
    def budget_used(self) -> float:
        """Cumulative budget consumed so far (iterations)."""
        return float(self._iterations)

    def record_iteration(self) -> None:
        """Advance the budget axis by one research-loop iteration."""
        self._iterations += 1

    def note_tool_call(self) -> None:
        """Count one tool call executed during the current iteration."""
        self._tool_calls += 1

    def note_sources(self, sources: Any) -> None:
        """Record the high-water mark of retrieved sources.

        ``sources`` may be the retriever's nested subquery buckets or a flat
        source list (mirrors the flattening used by the auditor and trace).
        """
        count = self._count_sources(sources)
        if count > self._sources:
            self._sources = count

    def record_reward(self, reward: float, budget: Optional[float] = None) -> None:
        """Record a reward observation at the current (or given) budget.

        Args:
            reward: Outcome quality in ``[0, 1]`` (e.g. grounding score).
            budget: Cumulative budget at which the reward was observed.
                Defaults to the budget consumed so far.
        """
        spent = self.budget_used if budget is None else float(budget)
        self._samples.append(EfficiencySample(budget=spent, reward=float(reward)))

    # -- reporting --------------------------------------------------------

    def samples(self) -> List[EfficiencySample]:
        """Recorded samples, in entry order (a copy, safe to iterate)."""
        return list(self._samples)

    def final_reward(self) -> float:
        """The most recently recorded reward (0.0 if none)."""
        return self._samples[-1].reward if self._samples else 0.0

    def auc(self) -> float:
        """AREK's normalized budget -> best-reward-so-far AUC in ``[0, 1]``.

        Samples are sorted by budget; the running maximum reward forms a
        non-decreasing step function over budget normalized to ``[0, 1]`` by the
        horizon. The AUC is the area under that step function. A run that
        reaches reward ``r`` using fraction ``b`` of the horizon scores
        ``r * (1 - b)``: high quality reached cheaply scores near ``r``; the
        same quality reached only at the horizon scores near ``0`` -- the
        efficiency dimension, distinct from outcome quality.
        """
        if not self._samples:
            return 0.0
        horizon = self.budget_horizon
        ordered = sorted(self._samples, key=lambda s: s.budget)
        area = 0.0
        running_max = 0.0
        prev_b = 0.0
        for sample in ordered:
            b = min(max(sample.budget / horizon, 0.0), 1.0)
            # On [prev_b, b] the best reward so far was the *previous* running max.
            area += running_max * (b - prev_b)
            running_max = max(running_max, max(sample.reward, 0.0))
            prev_b = b
        area += running_max * (1.0 - prev_b)
        return area

    def report(self) -> EfficiencyReport:
        """Build an :class:`EfficiencyReport` summarizing the run."""
        return EfficiencyReport(
            final_reward=self.final_reward(),
            budget_used=self.budget_used,
            budget_horizon=self.budget_horizon,
            auc=self.auc(),
            n_samples=len(self._samples),
            tool_calls=self._tool_calls,
            sources=self._sources,
            samples=self.samples(),
        )

    def render(self) -> str:
        """Render the efficiency dimension as a markdown report section."""
        rep = self.report()
        lines: List[str] = ["", "", "---", "", "## Search Efficiency", ""]
        lines.append(
            "> Budget-to-reward efficiency alongside outcome quality: how little"
        )
        lines.append(
            "> search budget (research-loop iterations) the run spent to reach"
        )
        lines.append(
            "> its grounding quality. Adapted from the efficiency measure in"
        )
        lines.append("> AREK (arXiv:2607.24647v1).")
        lines.append("")

        if not rep.verifiable:
            lines.append(
                "**Note:** no grounding score was captured for this report, so "
                "search efficiency could not be measured."
            )
            return "\n".join(lines)

        reward_pct = round(rep.final_reward * 100)
        budget_frac = rep.budget_used / rep.budget_horizon if rep.budget_horizon else 0.0
        auc_pct = round(rep.auc * 100)
        lines.append(
            f"**Outcome quality:** {reward_pct}% grounding score "
            f"(auditor reward)."
        )
        lines.append(
            f"**Budget consumed:** {rep.budget_used:.0f} of "
            f"{rep.budget_horizon:.0f} iterations "
            f"({round(budget_frac * 100)}%)"
            f"{_cost_detail(rep)}."
        )
        lines.append(
            f"**Search efficiency (AUC):** {auc_pct}% -- the area under the "
            f"budget-to-reward frontier. This is a *separate dimension* from "
            f"outcome quality: a well-grounded report that took most of the "
            f"budget still scores low here."
        )
        lines.append("")
        return "\n".join(lines)

    # -- internals --------------------------------------------------------

    @staticmethod
    def _count_sources(sources: Any) -> int:
        """Count flat sources, flattening WebSearchRetriever subquery buckets."""
        if not isinstance(sources, list):
            return 0
        total = 0
        for src in sources:
            if isinstance(src, dict) and ("results" in src or "similar_results" in src):
                for bucket_key in ("results", "similar_results"):
                    bucket = src.get(bucket_key) or []
                    total += len(bucket) if isinstance(bucket, list) else 0
            else:
                total += 1
        return total


def _cost_detail(rep: EfficiencyReport) -> str:
    """Human-readable supporting-cost suffix for the budget line."""
    parts: List[str] = []
    if rep.tool_calls:
        parts.append(f"{rep.tool_calls} tool call(s)")
    if rep.sources:
        parts.append(f"{rep.sources} source(s)")
    return f" ({', '.join(parts)})" if parts else ""
