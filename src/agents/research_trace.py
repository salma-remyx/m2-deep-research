"""Auditable trace of a research run (the "Graph of Trace").

Adapted from the **Graph of Trace** in *BrainPilot: Automating Brain Discovery
with Agentic Research* (arXiv:2607.15079v1). In BrainPilot every major step is
recorded in an auditable graph that links subgoals, tool use, evidence, and
claims so a researcher can follow and inspect exactly how a result was produced.

This is a **Mode 2 (adapted port)** of that mechanism for this deep-research
pipeline:

* BrainPilot's full graph over a PI agent and its brain-science specialists is
  replaced by a **per-step trace over this pipeline's actual agents** -- the
  user subgoal, each ``planning_agent`` / ``web_search_retriever`` tool call,
  the evidence those calls returned, and the final synthesized report.
* The record is **parameter-free and deterministic**: it captures what the
  supervisor already did (tool names, inputs, source counts, citation counts),
  so it needs no extra API calls and renders offline. It complements the
  :class:`~src.agents.auditor.ReportAuditor` grounding check rather than
  replacing it -- the trace shows *how* the report was built, the auditor
  checks *whether* its claims are grounded.

The trace is rendered as a ``## Graph of Trace`` section appended to the report,
so the workflow that produced it travels with the result and can be inspected.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Markdown inline citation: [label](url). Used to count claims backed by a link.
_LINK_RE = re.compile(r"\[[^\]]*\]\(https?://[^\s)]+\)")

# Node kinds, ordered by the workflow stage they represent.
SUBGOAL = "subgoal"
TOOL_USE = "tool_use"
EVIDENCE = "evidence"
CLAIM = "claim"

_KIND_LABELS: Dict[str, str] = {
    SUBGOAL: "Subgoal",
    TOOL_USE: "Tool",
    EVIDENCE: "Evidence",
    CLAIM: "Report",
}


@dataclass
class TraceNode:
    """A single recorded step in the research workflow."""

    node_id: int
    kind: str
    summary: str
    detail: str = ""
    parent: Optional[int] = None


@dataclass
class ResearchTrace:
    """Records the workflow that produced a report as a linked node graph.

    Nodes are appended in execution order and linked by ``parent`` so the
    rendered trace forms a tree: the root subgoal, the tool calls made to
    pursue it, the evidence each call returned, and the final report. The
    recorder is intentionally free of any model or network dependency -- it
    only captures metadata the supervisor already has on hand.
    """

    _nodes: List[TraceNode] = field(default_factory=list)
    _root: Optional[int] = None
    _last_tool: Optional[int] = None
    _last_evidence: Optional[int] = None

    @property
    def nodes(self) -> List[TraceNode]:
        """Recorded nodes in execution order (a copy, safe to iterate)."""
        return list(self._nodes)

    def reset(self) -> None:
        """Clear the trace so the recorder can be reused for a new run."""
        self._nodes = []
        self._root = None
        self._last_tool = None
        self._last_evidence = None

    def record(
        self,
        kind: str,
        summary: str,
        detail: str = "",
        parent: Optional[int] = None,
    ) -> int:
        """Append a node and return its id."""
        node = TraceNode(
            node_id=len(self._nodes) + 1,
            kind=kind,
            summary=summary.strip(),
            detail=detail.strip(),
            parent=parent,
        )
        self._nodes.append(node)
        return node.node_id

    # -- convenience recorders matching the supervisor's workflow ----------

    def record_subgoal(self, goal: str) -> int:
        """Record the top-level research subgoal (the user's query)."""
        self._root = self.record(SUBGOAL, goal)
        self._last_tool = None
        self._last_evidence = None
        return self._root

    def record_tool(self, name: str, tool_input: Any) -> int:
        """Record a tool invocation under the current subgoal."""
        self._last_tool = self.record(
            TOOL_USE, name, self._summarize_input(tool_input), parent=self._root
        )
        return self._last_tool

    def record_evidence(self, sources: Any) -> int:
        """Record evidence returned by the most recent tool call."""
        count = self._count_sources(sources)
        summary = f"{count} source(s) retrieved"
        self._last_evidence = self.record(
            EVIDENCE, summary, parent=self._last_tool or self._root
        )
        return self._last_evidence

    def record_report(self, report: str) -> int:
        """Record the synthesized report and how many claims it cites.

        The report is the terminus of the subgoal->tool->evidence->claim chain,
        so it is chained under the most recent evidence node (falling back to the
        last tool call, then the subgoal, if a report is recorded before any
        evidence was gathered).
        """
        n_citations = len(_LINK_RE.findall(report or ""))
        summary = "synthesized final report"
        detail = f"{n_citations} inline citation(s)"
        parent = self._last_evidence or self._last_tool or self._root
        return self.record(CLAIM, summary, detail, parent=parent)

    def render(self) -> str:
        """Render the trace as a markdown ``Graph of Trace`` section."""
        lines: List[str] = ["", "", "---", "", "## Graph of Trace", ""]
        lines.append(
            "> Auditable record of the research workflow -- every subgoal, tool"
        )
        lines.append(
            "> call, and piece of evidence that produced this report. Adapted"
        )
        lines.append("> from BrainPilot's Graph of Trace (arXiv:2607.15079v1).")
        lines.append("")

        if not self._nodes:
            lines.append("_No workflow steps were recorded for this report._")
            return "\n".join(lines)

        depths = self._depths()
        for node in self._nodes:
            indent = "  " * depths[node.node_id]
            label = _KIND_LABELS.get(node.kind, node.kind.title())
            text = f"{indent}- **[{node.node_id}] {label}:** {node.summary}"
            if node.detail:
                text += f" _({node.detail})_"
            lines.append(text)
        lines.append("")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _depths(self) -> Dict[int, int]:
        """Compute each node's indentation depth from its parent chain."""
        by_id = {n.node_id: n for n in self._nodes}
        depths: Dict[int, int] = {}
        for node in self._nodes:
            depth = 0
            parent = node.parent
            # Guard against cycles/missing parents; the graph is a shallow tree.
            seen = set()
            while parent is not None and parent in by_id and parent not in seen:
                seen.add(parent)
                depth += 1
                parent = by_id[parent].parent
            depths[node.node_id] = depth
        return depths

    def _summarize_input(self, tool_input: Any, max_chars: int = 120) -> str:
        """Produce a short, human-readable summary of a tool's input."""
        if isinstance(tool_input, dict):
            for key in ("research_query", "query", "subqueries_json"):
                value = tool_input.get(key)
                if value:
                    return self._truncate(str(value), max_chars)
            if tool_input:
                return self._truncate(str(next(iter(tool_input.values()))), max_chars)
            return ""
        return self._truncate(str(tool_input), max_chars) if tool_input else ""

    def _count_sources(self, sources: Any) -> int:
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

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        text = " ".join(text.split())
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
