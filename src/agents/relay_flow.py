"""Information-bottleneck coordination analyzer for the multi-agent relay chain.

Adapted from *When Do Multi-Agent Systems Help? An Information Bottleneck
Perspective* (arXiv:2607.16133v1). The paper's central observation is that a
multi-agent system (MAS) exchanges information only through **bounded relay**
messages between isolated agent contexts, so multi-agent design is an
information-bottleneck optimization: compressing redundant context out of the
relays improves *efficiency*, but may drop *task-relevant information*. MAS
helps when the **context-reduction benefit outweighs the relay information
loss** -- an effective parameter ``beta`` that shifts with model capability --
and MAS gains shrink or reverse once the relays become lossy.

This is a **Mode 2 (adapted port)** of that lens for this deep-research
pipeline:

* The paper's theoretical ``beta`` and its benchmark episode-accounting
  (``metrics.py``) are replaced by a **parameter-free, per-run diagnostic**
  over this pipeline's own relay chain -- supervisor -> planning_agent ->
  web_search_retriever -> synthesized report -- reconstructed from the
  :class:`~src.agents.research_trace.ResearchTrace` the supervisor already
  keeps. It makes no extra API calls and is fully deterministic.
* The paper's "infinite-bandwidth" single-agent baseline is replaced by a
  simple proxy: the **fraction of retrieved evidence that survives the bounded
  relays into the report** (evidence retention). When most gathered sources are
  never cited, the relays are lossy and the MAS pays for little gain, mirroring
  the paper's empirical finding that "MAS helps when relays are
  near-sufficient".

**The two axes are measured in genuinely different units, on purpose.** The
paper's trade-off is between *context size* (efficiency) and *task-relevant
information* (semantics); measuring both off the same retained/gathered count
would collapse them into one number and make ``beta`` identically zero. So:

* **Context-reduction benefit** is *character-based*: how much gathered source
  text the bounded relays kept out of the supervisor's shared context, relative
  to the delivered report. (Proxy: the paper's actual benefit is the
  transmission cost avoided; we approximate it with raw source-text size vs.
  report size.)
* **Information loss** is *evidence-unit-based*: the fraction of gathered
  sources never surfaced in the report.

It complements :class:`~src.agents.auditor.ReportAuditor` (which checks whether
the report's citations are *grounded* in retrieved sources -- the precision
direction) and :class:`~src.agents.research_trace.ResearchTrace` (which records
*how* the report was built): this analyzer asks the paper's question -- *did
the multi-agent relays help or hurt this run?* -- by measuring the recall
direction (how much gathered evidence flowed through).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from src.agents.research_trace import EVIDENCE, SUBGOAL

# Markdown inline citation: [label](url). Used to see which sources surfaced.
_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")


@dataclass
class RelayHop:
    """A single bounded relay between two agents/stages in the chain."""

    name: str
    producer: str
    consumer: str
    payload: str
    loss: Optional[float] = None  # fraction of information dropped, in [0, 1]
    reduction: Optional[float] = None  # fraction of context compressed, in [0, 1]
    note: str = ""


@dataclass
class CoordinationAnalysis:
    """Outcome of an information-bottleneck analysis over one research run."""

    model_name: str = ""
    n_subqueries: int = 0
    n_sources_gathered: int = 0
    n_sources_retained: int = 0
    gathered_text_chars: int = 0
    report_chars: int = 0
    retention: float = 0.0  # fraction of gathered sources cited in the report
    information_loss: float = 1.0  # 1 - retention
    reduction_ratio: float = 0.0  # gathered source chars per report char
    reduction_benefit: float = 0.0  # saturating compression, in [0, 1]
    effective_beta: float = 0.0  # reduction_benefit - information_loss, in [-1, 1]
    favorable: bool = False  # True when reduction outweighs loss
    verifiable: bool = True  # False when no evidence was gathered to judge
    hops: List[RelayHop] = field(default_factory=list)
    verdict: str = ""


class RelayFlowAnalyzer:
    """Quantify information flow across the MAS relay chain for a single run.

    The analyzer is parameter-free: given the research trace (the relay chain),
    the sources gathered by the retriever, and the synthesized report, it
    reconstructs the bounded relays and asks whether this run's context
    reduction outweighed its relay information loss.
    """

    def analyze(
        self,
        trace: Any,
        sources: Iterable[Any],
        report: str,
        model_name: str = "",
    ) -> CoordinationAnalysis:
        """Analyze the relay chain captured in ``trace`` against ``sources``.

        Args:
            trace: The :class:`~src.agents.research_trace.ResearchTrace` for the
                run; its nodes define the relay chain.
            sources: Sources gathered by the web search retriever. May be flat
                source dicts or the nested subquery buckets it emits.
            report: The synthesized report text (with inline citations).
            model_name: The supervisor model, surfaced as the capability proxy
                the paper's ``beta`` depends on.

        Returns:
            A :class:`CoordinationAnalysis` summarizing the relay trade-off.
        """
        flat = list(self._flatten_sources(sources))
        n_sources = len(flat)
        report = report or ""
        n_subqueries = self._count_subquery_buckets(sources)

        if n_sources == 0:
            # Nothing was gathered, so relay sufficiency cannot be judged.
            return CoordinationAnalysis(
                model_name=model_name,
                n_subqueries=n_subqueries,
                verifiable=False,
                information_loss=1.0,
                retention=0.0,
                hops=self._hops_from_trace(trace, n_subqueries, 0, 0, 0.0, report),
                verdict=(
                    "No evidence was gathered, so relay sufficiency cannot be "
                    "judged for this run."
                ),
            )

        gathered_urls = {
            self._normalize_url(src["url"])
            for src in flat
            if isinstance(src, dict) and src.get("url")
        }
        cited_urls = {
            self._normalize_url(url) for _, url in _LINK_RE.findall(report)
        }
        retained_urls = gathered_urls & cited_urls
        n_retained = len(retained_urls)

        retention = n_retained / n_sources
        information_loss = 1.0 - retention

        gathered_text_chars = sum(len(self._source_text(src)) for src in flat)
        report_chars = len(report)
        if gathered_text_chars > 0 and report_chars > 0:
            reduction_ratio = gathered_text_chars / report_chars
            reduction_benefit = max(
                0.0, (reduction_ratio - 1.0) / reduction_ratio
            )
        elif gathered_text_chars > 0:
            # Report is empty but sources carried text: maximal compression.
            reduction_ratio = float("inf")
            reduction_benefit = 1.0
        else:
            reduction_ratio = 0.0
            reduction_benefit = 0.0

        effective_beta = reduction_benefit - information_loss
        favorable = effective_beta > 0.0

        return CoordinationAnalysis(
            model_name=model_name,
            n_subqueries=n_subqueries,
            n_sources_gathered=n_sources,
            n_sources_retained=n_retained,
            gathered_text_chars=gathered_text_chars,
            report_chars=report_chars,
            retention=retention,
            information_loss=information_loss,
            reduction_ratio=reduction_ratio,
            reduction_benefit=reduction_benefit,
            effective_beta=effective_beta,
            favorable=favorable,
            verifiable=True,
            hops=self._hops_from_trace(
                trace, n_subqueries, n_sources, n_retained, reduction_benefit, report
            ),
            verdict=self._verdict(
                retention, reduction_benefit, information_loss, model_name
            ),
        )

    def format_report(self, analysis: CoordinationAnalysis) -> str:
        """Render an analysis as a markdown section to append to a report."""
        lines: List[str] = [
            "",
            "",
            "---",
            "",
            "## Multi-Agent Coordination Analysis",
            "",
        ]
        lines.append(
            "> Information-bottleneck view of this run's multi-agent relays:"
        )
        lines.append("> does context reduction outweigh relay information loss?")
        lines.append(
            "> Adapted from \"When Do Multi-Agent Systems Help? An Information"
        )
        lines.append("> Bottleneck Perspective\" (arXiv:2607.16133).")
        lines.append("")

        if not analysis.verifiable:
            lines.append(
                "**Note:** no retrieved evidence was captured for this report; "
                "relay sufficiency could not be judged."
            )
            lines.append("")
            lines.append(f"_Verdict:_ {analysis.verdict}")
            return "\n".join(lines)

        beta = analysis.effective_beta
        flag = "favorable" if analysis.favorable else "lossy"
        lines.append(
            f"**Coordination balance (effective beta = {beta:+.2f}):** {flag}. "
            "Positive means context reduction outweighed relay information loss."
        )
        lines.append("")

        lines.append("| Relay | Producer -> Consumer | Bounded payload | Metric |")
        lines.append("|-------|----------------------|-----------------|--------|")
        for hop in analysis.hops:
            metric = ""
            if hop.loss is not None:
                metric = f"loss {round(hop.loss * 100)}%"
            elif hop.reduction is not None:
                metric = f"reduction {round(hop.reduction * 100)}%"
            lines.append(
                f"| {hop.name} | {hop.producer} -> {hop.consumer} | "
                f"{hop.payload} | {metric} |"
            )
        lines.append("")

        subq_word = "subquery" if analysis.n_subqueries == 1 else "subqueries"
        pct_retain = round(analysis.retention * 100)
        pct_loss = round(analysis.information_loss * 100)
        pct_benefit = round(analysis.reduction_benefit * 100)
        lines.append(
            f"- **Evidence gathered:** {analysis.n_sources_gathered} source(s) "
            f"across {analysis.n_subqueries} {subq_word} "
            f"({analysis.gathered_text_chars} chars)."
        )
        lines.append(
            f"- **Evidence retained in report:** {analysis.n_sources_retained}/"
            f"{analysis.n_sources_gathered} ({pct_retain}%)."
        )
        lines.append(
            f"- **Context-reduction benefit:** {pct_benefit}% "
            "(chars the bounded relays kept out of the shared context)."
        )
        lines.append(
            f"- **Relay information loss:** {pct_loss}% of gathered evidence "
            "dropped before the report."
        )
        lines.append("")
        lines.append(f"_Verdict:_ {analysis.verdict}")
        lines.append("")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _verdict(
        self,
        retention: float,
        benefit: float,
        loss: float,
        model_name: str,
    ) -> str:
        if retention == 0.0:
            head = (
                "Relays dropped all gathered evidence before the report -- the "
                "multi-agent chain is information-lossy for this run."
            )
        elif benefit > loss:
            head = (
                "Multi-agent coordination looks favorable: context reduction "
                "outweighs relay information loss."
            )
        elif loss > benefit:
            head = (
                "Multi-agent coordination looks lossy: relay information loss "
                "exceeds the context-reduction benefit."
            )
        else:
            head = (
                "Multi-agent coordination is balanced: context reduction and "
                "relay information loss are even."
            )
        tail = (
            "Per the information-bottleneck view, MAS helps most when relays "
            "are near-sufficient (high retention)"
        )
        if model_name:
            tail += (
                f"; for a stronger supervisor like {model_name} the balance "
                "tilts further because it can already extract information from "
                "redundant context, so compression buys less"
            )
        tail += "."
        return f"{head} {tail}"

    def _hops_from_trace(
        self,
        trace: Any,
        n_subqueries: int,
        n_sources: int,
        retained: int,
        reduction_benefit: float,
        report: str,
    ) -> List[RelayHop]:
        """Reconstruct the bounded relays from the recorded trace nodes."""
        nodes: Sequence[Any] = list(getattr(trace, "nodes", []) or [])
        kinds = [getattr(n, "kind", "") for n in nodes]
        summaries = [getattr(n, "summary", "") for n in nodes]
        has_subgoal = SUBGOAL in kinds
        has_planning = any("planning_agent" in s for s in summaries)
        has_retriever = any("web_search_retriever" in s for s in summaries)
        has_evidence = EVIDENCE in kinds
        # The evidence->report relay exists whenever evidence was synthesized
        # into a report, whether or not record_report() was also called.
        has_report = bool(report)

        loss = (n_sources - retained) / n_sources if n_sources else None
        subq_word = "subquery" if n_subqueries == 1 else "subqueries"

        hops: List[RelayHop] = []
        if has_subgoal and has_planning:
            hops.append(
                RelayHop(
                    "R1",
                    "supervisor",
                    "planning_agent",
                    "research query",
                    note="decomposes the query; planner never sees the reasoning",
                )
            )
        if has_planning and has_retriever:
            hops.append(
                RelayHop(
                    "R2",
                    "planning_agent",
                    "web_search_retriever",
                    f"{n_subqueries} {subq_word} (bounded JSON)",
                    note="retriever sees only the decomposition",
                )
            )
        if has_retriever and has_evidence:
            hops.append(
                RelayHop(
                    "R3",
                    "web_search_retriever",
                    "supervisor",
                    f"{n_sources} source(s) -> bounded findings",
                    reduction=reduction_benefit,
                    note="full source text compressed into a synthesis",
                )
            )
        if has_evidence and has_report:
            hops.append(
                RelayHop(
                    "R4",
                    "supervisor",
                    "report",
                    f"{retained}/{n_sources} gathered source(s) cited",
                    loss=loss,
                    note="evidence -> claim relay",
                )
            )
        return hops

    def _flatten_sources(
        self, sources: Iterable[Any]
    ) -> Iterator[Dict[str, Any]]:
        """Yield flat source dicts, flattening retriever subquery buckets."""
        for src in sources or []:
            if not isinstance(src, dict):
                continue
            if "results" in src or "similar_results" in src:
                for bucket_key in ("results", "similar_results"):
                    for item in src.get(bucket_key) or []:
                        if isinstance(item, dict):
                            yield item
            else:
                yield src

    def _count_subquery_buckets(self, sources: Iterable[Any]) -> int:
        """Count subquery buckets (one per searched subquery); >=1 if flat."""
        src_list = list(sources or [])
        buckets = [
            s
            for s in src_list
            if isinstance(s, dict) and ("results" in s or "similar_results" in s)
        ]
        if buckets:
            return len(buckets)
        return 1 if any(isinstance(s, dict) for s in src_list) else 0

    def _source_text(self, src: Dict[str, Any]) -> str:
        """Concatenate a source's text-bearing fields into one string."""
        parts = [src.get("title") or "", src.get("text") or ""]
        highlights = src.get("highlights")
        if isinstance(highlights, list):
            parts.append(" ".join(str(h) for h in highlights))
        return " ".join(parts)

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize a URL for set comparison (drop scheme/www/fragment)."""
        cleaned = url.strip().lower().split("#")[0]
        cleaned = re.sub(r"^https?://(www\.)?", "", cleaned)
        return cleaned.rstrip("/")
