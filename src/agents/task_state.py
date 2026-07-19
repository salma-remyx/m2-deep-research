"""Unified task state with verifier-backed transitions.

Adapted (Mode 2) port of the state-centered design from StructAgent
(arXiv:2607.11388v1, "StructAgent: Harness Long-horizon Digital Agents
with Unified Causal Structure"). The paper's *core mechanism* is kept at
full fidelity:

  * a **unified state** capturing compact, verifiable task progress
    (typed milestones + bound evidence), and
  * **verifier-backed state transitions** -- progress is committed only
    when a verifier accepts the tool result, and rejected results are
    routed as attributed failures instead of looping blindly.

Auxiliary components substituted for this text-research pipeline:

  * The paper's learned / LLM verifier is replaced by a parameter-free
    heuristic verifier (:meth:`TaskState._verify`) that checks whether a
    tool result actually carries grounded evidence (a parseable plan,
    source URLs, non-error content of sufficient length). This
    approximates the "grounded in verification" signal without a second
    model call.
  * The OSWorld computer-use action space and its separate
    benchmark / eval harness are cut -- they do not map to a
    web-research pipeline, and evaluation belongs in a downstream PR.

The simplified **DONE auditor** (:meth:`TaskState.is_done`) is retained:
it judges whether enough *verified* evidence has accumulated to
synthesize a report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class MilestoneStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"


class Transition(str, Enum):
    """Outcome of a verifier-backed state transition."""

    VERIFIED = "verified"
    FAILED = "failed"


@dataclass
class Evidence:
    """A grounded piece of evidence bound to a verified tool result."""

    tool: str
    tool_use_id: str = ""
    url: str = ""
    snippet: str = ""


@dataclass
class Failure:
    """An attributed failure: which tool failed and why it was rejected."""

    tool: str
    tool_use_id: str = ""
    reason: str = ""


@dataclass
class Milestone:
    """A unit of verifiable task progress."""

    id: str
    description: str
    status: MilestoneStatus = MilestoneStatus.PENDING
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Verdict:
    verified: bool
    reason: str
    evidence: List[Evidence] = field(default_factory=list)


# Minimum grounded content for a tool result to count as verified.
_MIN_RESULT_CHARS = 120
_URL_RE = re.compile(r"https?://[^\s)\"']+", re.IGNORECASE)


class TaskState:
    """Unified, verifiable task-progress state for the research pipeline.

    The supervisor feeds each tool result through
    :meth:`commit_tool_result`; progress is committed only when the
    verifier accepts it, and rejected results are recorded as attributed
    failures. This closes the gap where ``research()`` previously looped
    to ``max_iterations`` with no progress / failure signal: the state
    always knows how much progress was *verified* and where failures
    were attributed.
    """

    def __init__(self) -> None:
        self._milestones: Dict[str, Milestone] = {}
        self._evidence: List[Evidence] = []
        self._failures: List[Failure] = []
        self.committed: int = 0
        self.total: int = 0

    # -- verifier (Mode 2 substitution for the paper's learned verifier) --

    def _verify(self, tool_name: str, result: str) -> Verdict:
        """Parameter-free verifier approximating StructAgent's verifier.

        Accepts a result iff it carries grounded, non-error evidence for
        the tool that produced it. No learned model is involved.
        """
        text = (result or "").strip()
        if not text:
            return Verdict(False, "empty tool result")
        # Tools in this pipeline surface failures as "Error ..." strings.
        if text.lower().startswith("error"):
            return Verdict(False, "tool returned an error")

        urls = _URL_RE.findall(text)
        if tool_name == "planning_agent":
            return self._verify_plan(text)
        if tool_name == "web_search_retriever":
            return self._verify_retrieval(text, urls)
        return self._verify_generic(text, urls)

    def _verify_plan(self, text: str) -> Verdict:
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return Verdict(False, "planning result is not valid JSON")
        subs = payload.get("subqueries") if isinstance(payload, dict) else None
        if not subs:
            return Verdict(False, "planning produced no subqueries")
        evidence = [Evidence("planning_agent", snippet=f"{len(subs)} subqueries")]
        return Verdict(True, f"{len(subs)} subqueries planned", evidence)

    def _verify_retrieval(self, text: str, urls: List[str]) -> Verdict:
        if len(text) < _MIN_RESULT_CHARS:
            return Verdict(False, "retrieval too short to carry evidence")
        if not urls:
            return Verdict(False, "retrieval produced no source URLs")
        evidence = [Evidence("web_search_retriever", url=u) for u in urls[:5]]
        return Verdict(True, f"{len(urls)} source URL(s)", evidence)

    def _verify_generic(self, text: str, urls: List[str]) -> Verdict:
        if len(text) < _MIN_RESULT_CHARS:
            return Verdict(False, "result too short to carry evidence")
        evidence = [Evidence("generic", url=u) for u in urls[:3]]
        return Verdict(True, "sufficient grounded content", evidence)

    # -- verifier-backed transition ----------------------------------------

    def commit_tool_result(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        result: str,
        tool_use_id: str = "",
    ) -> Transition:
        """Commit a tool result through the verifier.

        Verified results advance a typed milestone and bind their
        evidence; rejected results are recorded as attributed failures.
        Either way the state stays grounded in what the verifier
        accepted. Returns the resulting :class:`Transition`.
        """
        del tool_input  # accepted for the call-site signature; not used by the proxy verifier
        self.total += 1
        verdict = self._verify(tool_name, result)

        milestone = self._milestone_for(tool_name)
        if verdict.verified:
            self.committed += 1
            self._evidence.extend(verdict.evidence)
            milestone.evidence.extend(verdict.evidence)
            milestone.status = MilestoneStatus.VERIFIED
            return Transition.VERIFIED

        self._failures.append(Failure(tool_name, tool_use_id, verdict.reason))
        if milestone.status != MilestoneStatus.VERIFIED:
            milestone.status = MilestoneStatus.FAILED
        return Transition.FAILED

    def _milestone_for(self, tool_name: str) -> Milestone:
        if tool_name not in self._milestones:
            self._milestones[tool_name] = Milestone(
                id=tool_name,
                description=f"Evidence gathered via {tool_name}",
            )
        return self._milestones[tool_name]

    # -- DONE auditor + reporting ------------------------------------------

    def is_done(self) -> bool:
        """Simplified evidence-driven DONE auditor.

        True once enough *verified* evidence has accumulated to
        synthesize a report: a verified planning result and a verified
        retrieval result carrying at least one source URL.
        """
        verified = {
            m.id
            for m in self._milestones.values()
            if m.status == MilestoneStatus.VERIFIED
        }
        has_sources = any(e.url for e in self._evidence)
        return (
            "planning_agent" in verified
            and "web_search_retriever" in verified
            and has_sources
        )

    def progress_report(self) -> str:
        """State-grounded summary used when the research loop is exhausted."""
        lines = [
            f"Research loop ended with {self.committed}/{self.total} "
            f"verified tool result(s)."
        ]
        for milestone in self._milestones.values():
            lines.append(f"  - [{milestone.status.value}] {milestone.id}")
        if self.evidence_urls:
            lines.append(
                f"  - {len(self.evidence_urls)} grounded source URL(s) bound."
            )
        if self._failures:
            lines.append("  - Attributed failures:")
            for failure in self._failures:
                lines.append(f"      * {failure.tool}: {failure.reason}")
        return "\n".join(lines)

    # -- read-only public views (for logging / tests) ----------------------

    @property
    def failures(self) -> List[Failure]:
        return list(self._failures)

    @property
    def evidence_urls(self) -> List[str]:
        return [e.url for e in self._evidence if e.url]
