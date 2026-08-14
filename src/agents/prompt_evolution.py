"""Evidence-guided evolution of agent prompts across research runs.

Adapted from **EMAS** (*Evolving Multi-Agent System*, arXiv:2608.07196v1).
EMAS turns the traces a multi-agent system already produces into structured
diagnoses, and only when the *same* diagnosis recurs across samples does it
propose a single revision (an operation applied to a named target), which is
then validated in pairs against the current system before it is kept. The
system improves without touching any LLM parameters -- revisions land as
prompt edits.

This is a **Mode 2 (adapted port)** of that loop for this deep-research
pipeline:

* EMAS's LLM diagnosis step is replaced by a **parameter-free diagnosis**
  derived from the run's own grounding audit and Graph of Trace
  (:class:`~src.agents.auditor.ReportAuditor` /
  :class:`~src.agents.research_trace.ResearchTrace`). Each diagnosis names a
  revision operation, a target, and the evidence count backing it -- the same
  (operation, target, evidence) triple EMAS's structured diagnoses carry.
* EMAS's revision *generator* (an LLM writing a new prompt) is replaced by a
  small library of **template revisions**, each a fixed instruction appended
  to the target agent's system prompt. The candidate revision is emitted only
  once a diagnosis has recurred ``min_recurrence`` times, mirroring EMAS's
  recurrence gate.
* EMAS's *paired validation* (re-running both systems on held-out samples
  with an accuracy scorer) is replaced by the pipeline's own deterministic
  grounding score: a candidate is accepted only if the next run's grounding
  score does not drop. There is no labelled benchmark here, so grounding is
  the natural acceptance criterion.

Revision state persists to a JSON file so evolution survives across process
runs.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Diagnosis kinds, keyed by (operation, target). Each entry carries the
# evidence field of the diagnosis and the template revision EMAS would apply.
from src.agents.research_trace import ResearchTrace

# The prompt-bearing agents EMAS can revise in this pipeline.
TARGET_PLANNING = "planning_agent"

# Diagnosis code -> (target, template revision). A diagnosis fires on a
# recurring failure mode; the revision is the countermeasure appended to the
# target agent's system prompt. Template fields ({n}, {k}) come from the
# diagnosis evidence.
_DIAGNOSIS_LIBRARY: Dict[str, Dict[str, Any]] = {
    "ungrounded_claims": {
        "target": TARGET_PLANNING,
        "revision": (
            "\n\nEvolution note ({n} prior runs produced reports with claims "
            "no retrieved source supports): before finalizing, verify each "
            "numeric claim and citation against the retrieved findings and "
            "drop any you cannot trace to a source.\n"
        ),
    },
    "fabricated_citations": {
        "target": TARGET_PLANNING,
        "revision": (
            "\n\nEvolution note ({n} prior runs cited URLs that were never "
            "retrieved): only cite URLs that appeared verbatim in the "
            "web_search_retriever findings; never compose or guess a source "
            "link.\n"
        ),
    },
    "thin_evidence": {
        "target": TARGET_PLANNING,
        "revision": (
            "\n\nEvolution note (prior runs gathered {k} or fewer sources per "
            "subquery): favour broad, self-contained subqueries and request "
            "more results per subquery so synthesis has enough evidence to "
            "draw on.\n"
        ),
    },
    "no_evidence": {
        "target": TARGET_PLANNING,
        "revision": (
            "\n\nEvolution note (prior runs returned no usable sources): "
            "reformulate subqueries as plain natural-language questions, drop "
            "overly restrictive domain filters, and retry before "
            "synthesizing.\n"
        ),
    },
}


@dataclass
class Revision:
    """A candidate or accepted prompt revision (EMAS's one-revision unit)."""

    diagnosis: str
    target: str
    instruction: str
    status: str = "candidate"  # candidate -> accepted | rejected
    evidence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "target": self.target,
            "instruction": self.instruction,
            "status": self.status,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Revision":
        return cls(
            diagnosis=data["diagnosis"],
            target=data["target"],
            instruction=data["instruction"],
            status=data.get("status", "candidate"),
            evidence=data.get("evidence", 0),
        )


@dataclass
class EvolutionState:
    """Accumulated evidence and revisions across runs (EMAS's memory)."""

    # Recurrence counts per diagnosis code -- EMAS accumulates evidence and
    # only proposes a revision once the same diagnosis recurs.
    diagnosis_counts: Dict[str, int] = field(default_factory=dict)
    revisions: List[Revision] = field(default_factory=list)
    # Grounding score of the most recent run that used an accepted revision,
    # used as the paired-validation baseline.
    last_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis_counts": self.diagnosis_counts,
            "revisions": [r.to_dict() for r in self.revisions],
            "last_score": self.last_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionState":
        return cls(
            diagnosis_counts=dict(data.get("diagnosis_counts", {})),
            revisions=[Revision.from_dict(r) for r in data.get("revisions", [])],
            last_score=data.get("last_score"),
        )


class PromptEvolution:
    """Diagnose recurring run failures and evolve agent prompts in response.

    The supervisor calls :meth:`observe_run` after each report (with the
    audit result and the run's trace) and :meth:`apply_accepted` before the
    next run's tool calls. Together they implement EMAS's
    Diagnose -> Accumulate-Evidence -> Propose-One-Revision ->
    Paired-Validation loop, parameter-free.
    """

    def __init__(
        self,
        state_path: Optional[str] = None,
        min_recurrence: int = 2,
        acceptance_drop: float = 0.0,
    ):
        self.state_path = state_path
        self.min_recurrence = min_recurrence
        self.acceptance_drop = acceptance_drop
        self.state = self._load()

    # -- EMAS step 1+2: diagnose and accumulate evidence --------------------

    def diagnose(self, audit: Any, trace: Optional[ResearchTrace] = None) -> List[str]:
        """Return the diagnosis codes this run's evidence supports.

        Args:
            audit: An :class:`~src.agents.auditor.AuditResult` for the run's
                report (may be ``None`` if auditing was skipped).
            trace: The run's :class:`~src.agents.research_trace.ResearchTrace`,
                used for the evidence-volume diagnoses.

        Returns:
            Sorted list of diagnosis codes (empty when the run was clean or
            unverifiable).
        """
        codes: List[str] = []
        if audit is not None and getattr(audit, "verifiable", False):
            if getattr(audit, "unsupported_citations", None):
                codes.append("fabricated_citations")
            if getattr(audit, "unsupported_claims", None):
                codes.append("ungrounded_claims")

        if trace is not None:
            source_counts = [
                n.summary.split()[0]
                for n in trace.nodes
                if n.kind == "evidence" and n.summary.startswith(tuple("0123456789"))
            ]
            counts = [int(c) for c in source_counts if c.isdigit()]
            if counts and max(counts) <= 2:
                codes.append("thin_evidence")
            elif not counts:
                codes.append("no_evidence")

        return sorted(set(codes))

    def observe_run(
        self, audit: Any, trace: Optional[ResearchTrace] = None
    ) -> Optional[Revision]:
        """Record one run's evidence; propose a revision on recurrence.

        This is EMAS's accumulate-evidence + propose-one-revision step: each
        diagnosis's count is incremented, and the first diagnosis to reach
        ``min_recurrence`` without an existing revision yields a candidate.
        """
        codes = self.diagnose(audit, trace)
        for code in codes:
            self.state.diagnosis_counts[code] = (
                self.state.diagnosis_counts.get(code, 0) + 1
            )

        proposed = None
        for code in codes:
            if self._revision_for(code) is not None:
                continue
            if self.state.diagnosis_counts[code] < self.min_recurrence:
                continue
            proposed = self._propose(code)
            if proposed is not None:
                self.state.revisions.append(proposed)
                break

        self.save()
        return proposed

    def _propose(self, code: str) -> Optional[Revision]:
        """Build the candidate revision for a recurring diagnosis."""
        entry = _DIAGNOSIS_LIBRARY.get(code)
        if entry is None:
            return None
        instruction = entry["revision"].format(
            n=self.state.diagnosis_counts.get(code, 0), k=2
        )
        return Revision(
            diagnosis=code,
            target=entry["target"],
            instruction=instruction,
            status="candidate",
            evidence=self.state.diagnosis_counts.get(code, 0),
        )

    def _revision_for(self, code: str) -> Optional[Revision]:
        for rev in self.state.revisions:
            if rev.diagnosis == code:
                return rev
        return None

    # -- EMAS step 3: paired validation -------------------------------------

    def validate(self, score: Optional[float]) -> None:
        """Accept/reject pending candidates against the observed score.

        EMAS validates a candidate revision by comparing the revised system
        against the current one under the acceptance criterion. Here the
        criterion is the pipeline's own grounding score: a candidate is
        accepted when there is no baseline yet (first revised run) or the
        score did not drop by more than ``acceptance_drop``, and rejected
        otherwise -- a rejected revision is retired so its diagnosis may
        propose a different one later.
        """
        pending = [r for r in self.state.revisions if r.status == "candidate"]
        if not pending:
            self.state.last_score = score if score is not None else self.state.last_score
            self.save()
            return

        baseline = self.state.last_score
        accept = baseline is None or (
            score is not None and score >= baseline - self.acceptance_drop
        )
        for rev in pending:
            rev.status = "accepted" if accept else "rejected"
        if score is not None:
            self.state.last_score = score
        self.save()

    # -- applying revisions --------------------------------------------------

    def apply_accepted(
        self, target_prompt: str, target: str, include_candidates: bool = False
    ) -> str:
        """Return ``target_prompt`` with the revisions selected for ``target``.

        Revisions already present in the prompt are skipped, so applying
        twice is a no-op. With ``include_candidates`` the still-pending
        candidates are applied too, so the revised system actually runs
        before :meth:`validate` judges it -- the target-native stand-in for
        EMAS's paired validation.
        """
        wanted = {"accepted"} | ({"candidate"} if include_candidates else set())
        prompt = target_prompt
        for rev in self.state.revisions:
            if rev.status not in wanted or rev.target != target:
                continue
            if rev.instruction.strip() in prompt:
                continue
            prompt = prompt + rev.instruction
        return prompt

    def pending(self) -> List[Revision]:
        """Revisions awaiting validation (for logging/inspection)."""
        return [r for r in self.state.revisions if r.status == "candidate"]

    # -- persistence ----------------------------------------------------------

    def _load(self) -> EvolutionState:
        if not self.state_path or not os.path.exists(self.state_path):
            return EvolutionState()
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                return EvolutionState.from_dict(json.load(fh))
        except (OSError, ValueError, KeyError, TypeError):
            # A corrupt state file must never break a research run.
            return EvolutionState()

    def save(self) -> None:
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(self.state.to_dict(), fh, indent=2)
        except OSError:
            pass

    def reset(self) -> None:
        """Clear all accumulated evidence and revisions."""
        self.state = EvolutionState()
        self.save()
