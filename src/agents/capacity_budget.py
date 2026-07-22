"""Per-role capacity budget for a hierarchical research run.

Adapted from the capacity-distribution analysis in *Think Big, Search Small:
Where Capacity Matters in Hierarchical Search Agents?* (arXiv:2607.07548v1).
That paper factorizes hierarchical search into a **delegation** role (task
decomposition / coordination) and an **execution** role (retrieval and
evidence extraction), holds the answer-generation role fixed, and sweeps model
capacity along each axis. Its central finding is that capacity sensitivity is
**asymmetric**: scaling the delegation backbone moves exact match by ~11
points, whereas scaling the execution sub-agent moves it by only ~2.6 points,
so delegation is the capability bottleneck. The prescribed recipe is therefore
*concentrate capacity at delegation and downsize execution without sacrificing
accuracy* -- and a 1.7B executor trained by quality-filtered trajectory
distillation matches a frontier sub-agent at 37% fewer sub-agent tokens.

This is a **Mode 2 (adapted port)** of that analysis for this deep-research
pipeline, which already splits into the paper's roles:

* **delegation** -- :class:`~src.agents.supervisor.SupervisorAgent` on
  MiniMax-M2.1, which decomposes the task, orchestrates the sub-agents, and
  synthesizes the final report (it also holds the answer-generation role,
  exactly as the paper holds the answerer fixed as a high-capacity constant).
* **execution** -- :class:`~src.agents.planning_agent.PlanningAgent` and
  :class:`~src.agents.web_search_retriever.WebSearchRetriever`, both on
  OpenRouter Gemini 2.5 Flash, which perform the sub-query decomposition,
  retrieval, and evidence extraction.

The following components are substituted or cut (Mode 2 honesty):

* The paper's **controlled capacity sweep across six model scales on five
  multi-hop QA benchmarks** (its experiment harness and exact-match scoring)
  is **cut** -- that is a standalone benchmark suite and evaluation belongs in
  a downstream PR. What survives is the *analysis shape*: measuring where
  compute is distributed across the two roles in a real run.
* The paper's **empirical capacity-sensitivity curve** (the learned +11 / +2.6
  EM deltas, which require the benchmark sweep) is replaced by the paper's own
  *reported* sensitivities, applied as constants to the recommendation rather
  than re-measured. No learned estimator is introduced.
* The paper's **1.7B distilled executor** (finding #3, which requires
  quality-filtered trajectory distillation training) is **out of scope** -- a
  training pipeline this repo cannot host. What is ported is the *recipe it
  implies*: execution can be downsized, and the budget quantifies the token
  opportunity using the paper's reported 37% sub-agent-token reduction as the
  Pareto anchor.

The core mechanism is preserved: an accounting pass that attributes compute
(token spend) to the delegation vs execution roles, reports the distribution,
and applies the paper's "downsize execution -- it is not the bottleneck"
recipe. It is parameter-free and deterministic -- it only consumes usage
metadata the supervisor and its sub-agents already produce, so it adds no API
calls and can run on every report.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

# Role labels, matching the paper's factorization.
DELEGATION = "delegation"
EXECUTION = "execution"

# The paper's reported capacity sensitivities (exact-match points per role
# when that role's backbone is scaled) and its Pareto token reduction. Used as
# constants in the recommendation -- they are the paper's own measurements, not
# re-estimated here.
_DELEGATION_EM_GAIN = 11.0
_EXECUTION_EM_GAIN = 2.6
_PARETO_TOKEN_REDUCTION = 0.37


@dataclass
class RoleSpend:
    """Compute attributed to one role over a research run."""

    role: str
    model: str = "unregistered"
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) consumed by this role."""
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate one call's usage into this role's totals."""
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.calls += 1


@dataclass
class CapacityDistribution:
    """Observed compute split between the delegation and execution roles."""

    delegation: RoleSpend
    execution: RoleSpend

    @property
    def total_tokens(self) -> int:
        return self.delegation.total_tokens + self.execution.total_tokens

    def share(self, role: str) -> float:
        """Fraction of total tokens attributable to ``role`` (0.0 if none)."""
        total = self.total_tokens
        if total <= 0:
            return 0.0
        spend = self.delegation if role == DELEGATION else self.execution
        return spend.total_tokens / total

    @property
    def delegation_share(self) -> float:
        return self.share(DELEGATION)

    @property
    def execution_share(self) -> float:
        return self.share(EXECUTION)

    @property
    def matches_recipe(self) -> bool:
        """True when delegation dominates compute (the 'think big' structure).

        The paper prescribes concentrating capacity at delegation. When the
        observed distribution already does so, the run matches the recipe; when
        execution out-spends delegation, execution is the lever to pull.
        """
        return self.delegation.total_tokens >= self.execution.total_tokens


@dataclass
class DownsizeRecommendation:
    """The paper's 'downsize execution' verdict for this run."""

    downsize_execution: bool
    rationale: str
    execution_output_tokens: int = 0
    estimated_token_saving: int = 0
    paper_reference: str = "arXiv:2607.07548v1"


class CapacityBudget:
    """Account compute across the delegation and execution roles.

    The supervisor records each role's token usage as its calls land
    (delegation usage comes from the MiniMax response, execution usage from the
    OpenRouter sub-agents). The budget then reports the distribution and
    applies the paper's recipe: because execution is not the capacity
    bottleneck, any measured execution output spend is a downsizing
    opportunity, quantified against the paper's reported 37% sub-agent-token
    reduction at matched accuracy.
    """

    def __init__(self) -> None:
        self._spend: Dict[str, RoleSpend] = {}

    def register(self, role: str, model: str) -> None:
        """Declare a role and the model that fills it."""
        self._spend[role] = RoleSpend(role=role, model=model or "unregistered")

    def reset(self) -> None:
        """Clear recorded usage so the budget can be reused for a new run.

        Role registrations (model assignments) are preserved.
        """
        for role, spend in list(self._spend.items()):
            self._spend[role] = RoleSpend(role=role, model=spend.model)

    def record_delegation(self, usage: Any) -> None:
        """Record one delegation (MiniMax) call's usage."""
        self._record(DELEGATION, usage)

    def record_execution(self, usage: Any) -> None:
        """Record one execution (OpenRouter sub-agent) call's usage."""
        self._record(EXECUTION, usage)

    def distribution(self) -> CapacityDistribution:
        """Return the observed delegation/execution compute split."""
        return CapacityDistribution(
            delegation=self._spend.get(DELEGATION, RoleSpend(role=DELEGATION)),
            execution=self._spend.get(EXECUTION, RoleSpend(role=EXECUTION)),
        )

    def recommend_downsize(self) -> DownsizeRecommendation:
        """Apply the paper's 'downsize execution' recipe to this run."""
        dist = self.distribution()
        exec_output = dist.execution.output_tokens
        if exec_output <= 0:
            return DownsizeRecommendation(
                downsize_execution=False,
                rationale=(
                    "No execution token spend was measured this run, so no "
                    "downsizing opportunity could be quantified. Instrument "
                    "the planning and retrieval sub-agents to obtain one."
                ),
            )

        saving = round(exec_output * _PARETO_TOKEN_REDUCTION)
        return DownsizeRecommendation(
            downsize_execution=True,
            rationale=(
                f"Execution (retrieval + evidence extraction) is not the "
                f"capacity bottleneck: scaling it moves EM ~{_EXECUTION_EM_GAIN} "
                f"pts vs ~{_DELEGATION_EM_GAIN} pts for delegation. Per the "
                f"recipe, downsize the execution role -- a smaller executor "
                f"matches a frontier sub-agent at ~37% fewer sub-agent tokens."
            ),
            execution_output_tokens=exec_output,
            estimated_token_saving=saving,
        )

    def render(self) -> str:
        """Render the budget as a markdown ``Capacity Budget`` section."""
        dist = self.distribution()
        rec = self.recommend_downsize()
        lines = [
            "",
            "",
            "---",
            "",
            "## Capacity Budget",
            "",
            "> Where compute was distributed across this run's delegation and",
            "> execution roles, and whether execution should be downsized.",
            "> Adapted from *Think Big, Search Small* (arXiv:2607.07548v1).",
            "",
        ]

        lines.append("| Role | Model | Calls | Input tok | Output tok | Total |")
        lines.append("|------|-------|-------|-----------|------------|-------|")
        for spend in (dist.delegation, dist.execution):
            lines.append(
                f"| {spend.role} | `{spend.model}` | {spend.calls} | "
                f"{spend.input_tokens} | {spend.output_tokens} | "
                f"{spend.total_tokens} |"
            )
        lines.append("")

        total = dist.total_tokens
        if total > 0:
            lines.append(
                f"**Distribution:** delegation {_pct(dist.delegation_share)} / "
                f"execution {_pct(dist.execution_share)} of {total} tokens."
            )
            if dist.matches_recipe:
                lines.append(
                    "Capacity is concentrated at delegation -- the 'think big' "
                    "structure the paper recommends."
                )
            else:
                lines.append(
                    "Execution is out-spending delegation -- the paper flags "
                    "execution as the lever to downsize."
                )
        else:
            lines.append("_No token usage was recorded for this run._")
        lines.append("")

        lines.append("### Downsizing recommendation (execution role)")
        lines.append(rec.rationale)
        if rec.downsize_execution:
            lines.append(
                f"**Estimated saving:** ~{rec.estimated_token_saving} sub-agent "
                f"output tokens ({rec.execution_output_tokens} measured "
                f"× {_PARETO_TOKEN_REDUCTION:.2f}, the paper's Pareto point "
                "at matched accuracy)."
            )
        lines.append("")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _record(self, role: str, usage: Any) -> None:
        if role not in self._spend:
            self._spend[role] = RoleSpend(role=role)
        input_tokens, output_tokens = self._usage_tokens(usage)
        self._spend[role].add(input_tokens, output_tokens)

    @staticmethod
    def _usage_tokens(usage: Any) -> Tuple[int, int]:
        """Extract (input, output) tokens from Anthropic or OpenRouter usage.

        Accepts the Anthropic SDK ``Usage`` object (``input_tokens`` /
        ``output_tokens`` attributes) or an OpenRouter usage dict
        (``prompt_tokens`` / ``completion_tokens``). ``None`` yields (0, 0).
        """
        if usage is None:
            return 0, 0
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is None and isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = (
                usage.get("completion_tokens") or usage.get("output_tokens")
            )
        return int(input_tokens or 0), int(output_tokens or 0)


def _pct(share: float) -> str:
    return f"{round(share * 100)}%"
