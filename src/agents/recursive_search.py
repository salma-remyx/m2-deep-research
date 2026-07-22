"""Recursive deep-and-wide web-search delegation.

Adapted from **WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide
Web Search** (arXiv:2607.08662v1). WebSwarm progressively instantiates agentic
search nodes, each coupling a local objective with a *search mode* (atom,
wide, deep, entity_collect). A node either solves its objective itself or
delegates child nodes for the evidence gaps it surfaces, then returns evidence
upward so parents can expand, revise, or aggregate. WebSwarm first probes how
task-relevant information is organized on the web to ground that expansion.

This is a **Mode 2 (adapted port)** for this pipeline's Exa-backed search,
which is otherwise single-pass and wide-only:

* WebSwarm's LLM-driven delegation estimator (the component that decides
  *where* to recurse) is replaced by a **parameter-free gap heuristic** --
  named entities and source domains surfaced in gathered results that no
  existing node covers. The recursive delegation tree, the four search modes,
  the web-structure probe, and the upward evidence aggregation are kept at
  full fidelity. The substitution mirrors how ``auditor.py`` replaced
  BrainPilot's LLM fabrication judge with a parameter-free grounding proxy:
  deterministic, needs no extra API keys, and is unit-testable offline.
* WebSwarm's "process-level experience reuse across homogeneous sibling
  nodes" is **not** ported -- it is an efficiency optimization, not part of
  the core recursive-delegation mechanism.

Output mirrors ``WebSearchRetriever.search_with_subqueries`` buckets (each
``subquery`` / ``priority`` / ``results`` / ``similar_results``), so the
existing ``retrieve() -> findings`` contract, the supervisor's source capture
into ``last_search_results``, and the ``ReportAuditor`` grounding pass keep
working unchanged. Each bucket additionally records the ``mode`` used, the
recursion ``depth`` reached, and the ``children`` objectives it delegated to.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple


# WebSwarm's four search modes. Each maps to a concrete ExaTool usage.
MODE_ATOM = "atom"  # precise single-shot lookup of one fact
MODE_WIDE = "wide"  # broad coverage of an objective
MODE_DEEP = "deep"  # drill into a top hit plus its neighbors
MODE_ENTITY = "entity_collect"  # gather everything about a named entity

# Named-entity probe: 1-4 Capitalized words (proper nouns / named things).
# Used only to spot evidence gaps to delegate on -- never as a precision filter.
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}\b")

# Capitalized tokens that are sentence mechanics, not entities worth delegating.
_GENERIC_ENTITIES: Set[str] = frozenset(
    """
    The This These Those That We Our Ours They Them Their It Its An A And But Or
    Nor So However According Recent Currently Reportedly Said Says Source
    Sources Figure Table Section Chapter Abstract Introduction Conclusion
    Summary Monday Tuesday Wednesday Thursday Friday Saturday Sunday January
    February March April May June July August September October November December
    """.split()
)


@dataclass
class _Budget:
    """Shared counter capping total Exa calls across the delegation tree.

    Mirrors WebSwarm's web-tool-efficiency constraint: recursion is bounded by a
    total search budget, not just by depth, so a wide root cannot fan out
    unbounded.
    """

    limit: int = 12
    calls: int = 0

    def take(self) -> bool:
        """Reserve one Exa call; return False (reserving nothing) if exhausted."""
        if self.calls >= self.limit:
            return False
        self.calls += 1
        return True


class RecursiveSearchDelegator:
    """Recursive delegation tree over Exa search (WebSwarm, arXiv:2607.08662v1).

    Each planned subquery seeds a root node; nodes run their search mode, then
    delegate ``entity_collect`` child nodes for named-entity gaps in their
    results, up to ``max_depth`` recursion levels and ``node_budget`` Exa calls.
    Evidence aggregates and deduplicates upward into
    ``search_with_subqueries``-shaped buckets.
    """

    def __init__(
        self,
        exa: Any,
        max_depth: int = 2,
        node_budget: int = 12,
        max_children: int = 3,
        min_entity_freq: int = 2,
    ):
        self.exa = exa
        self.max_depth = max_depth
        self.node_budget = node_budget
        self.max_children = max_children
        self.min_entity_freq = min_entity_freq

    def delegate(
        self,
        research_query: str,
        subqueries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run a recursive delegation forest over the planned subqueries.

        Args:
            research_query: The original research query; probed first to ground
                which entities later nodes collect.
            subqueries: Planned subqueries (``query`` / ``type`` / ``priority``).

        Returns:
            One evidence bucket per subquery (plus any probe-grounded entity
            nodes), in ``WebSearchRetriever.search_with_subqueries`` shape.
        """
        budget = _Budget(limit=self.node_budget)
        probe = self._probe_structure(research_query, budget)

        # ``covered`` tracks objectives that already have a node, so siblings do
        # not duplicate one another's work. Seed it with the root query and the
        # planned subquery objectives.
        covered: Set[str] = {research_query.lower()}
        for subquery in subqueries:
            objective = (subquery.get("query") or "").strip().lower()
            if objective:
                covered.add(objective)

        # Probe-grounded entity_collect nodes come first, so the planned nodes
        # see those entities as already covered and do not re-collect them.
        probe_buckets: List[Dict[str, Any]] = []
        for entity in probe["entities"]:
            if budget.calls >= budget.limit:
                break
            key = entity.lower()
            if key in covered:
                continue
            covered.add(key)
            probe_buckets.append(
                self._solve_node(entity, MODE_ENTITY, 4, 0, budget, covered)
            )

        planned_buckets: List[Dict[str, Any]] = []
        for index, subquery in enumerate(subqueries):
            objective = (subquery.get("query") or "").strip()
            if not objective:
                continue
            mode = self._mode_for_subquery(subquery)
            bucket = self._solve_node(
                objective, mode, subquery.get("priority", 3), 0, budget, covered
            )
            # Fold the structure probe's evidence into the first planned bucket
            # so the probe search is not wasted, deduped against its results.
            if index == 0 and probe["results"]:
                bucket["results"] = self._dedupe(probe["results"] + bucket["results"])
            planned_buckets.append(bucket)

        return planned_buckets + probe_buckets

    # -- recursive core ---------------------------------------------------

    def _solve_node(
        self,
        objective: str,
        mode: str,
        priority: int,
        depth: int,
        budget: _Budget,
        covered: Set[str],
    ) -> Dict[str, Any]:
        """Solve one node: run its search mode, then delegate children for gaps."""
        results, similar = self._run_mode(objective, mode, budget)
        bucket: Dict[str, Any] = {
            "subquery": objective,
            "priority": priority,
            "results": results,
            "similar_results": similar,
            "mode": mode,
            "depth": depth,
            "children": [],
        }
        covered.add(objective.lower())

        if depth < self.max_depth and budget.calls < budget.limit:
            gaps = self._detect_gaps(results + similar, covered)
            for gap in gaps[: self.max_children]:
                if budget.calls >= budget.limit:
                    break
                child = self._solve_node(
                    gap, MODE_ENTITY, priority, depth + 1, budget, covered
                )
                bucket["results"] += child["results"]
                bucket["similar_results"] += child["similar_results"]
                bucket["children"].append(child["subquery"])

        bucket["results"] = self._dedupe(bucket["results"])
        bucket["similar_results"] = self._dedupe(bucket["similar_results"])
        return bucket

    def _run_mode(
        self, objective: str, mode: str, budget: _Budget
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute a node's search mode against Exa; return (results, similar)."""
        results: List[Dict[str, Any]] = []
        similar: List[Dict[str, Any]] = []
        if mode == MODE_ATOM:
            if budget.take():
                results = self._search(objective, 3)
        elif mode in (MODE_DEEP, MODE_ENTITY):
            if budget.take():
                results = self._search(objective, 10 if mode == MODE_ENTITY else 8)
            if results and budget.take():
                similar = self._find_similar(results[0], 5)
        else:  # MODE_WIDE
            if budget.take():
                results = self._search(objective, 15)
        return results, similar

    # -- web-structure probe (WebSwarm step 1) ----------------------------

    def _probe_structure(
        self, research_query: str, budget: _Budget
    ) -> Dict[str, Any]:
        """Probe how task-relevant information is organized on the web.

        One wide search surfaces dominant source domains and named entities,
        grounding which entities later nodes collect. Parameter-free: regex
        entity extraction plus domain frequency -- no learned estimator.
        """
        probe: Dict[str, Any] = {"domains": [], "entities": [], "results": []}
        if not research_query or not budget.take():
            return probe
        try:
            raw = self.exa.search(query=research_query, num_results=5)
            results = self.exa.format_results(raw)
        except Exception:
            return probe
        probe["results"] = results

        domain_freq: Dict[str, int] = {}
        for result in results:
            host = result.get("url", "")
            match = re.match(r"https?://([^/]+)", host)
            if match:
                domain = match.group(1).lower().lstrip("www.")
                domain_freq[domain] = domain_freq.get(domain, 0) + 1
        probe["domains"] = [
            domain
            for domain, _ in sorted(domain_freq.items(), key=lambda kv: -kv[1])
        ][:5]
        probe["entities"] = self._detect_gaps(results, {research_query.lower()})
        return probe

    # -- gap detection (parameter-free delegation estimator) --------------

    def _detect_gaps(
        self, results: List[Dict[str, Any]], covered: Set[str]
    ) -> List[str]:
        """Return named entities in ``results`` no node has covered yet.

        Substitutes for WebSwarm's LLM delegation estimator: a deterministic
        entity-frequency heuristic over gathered titles, text, and highlights.
        """
        frequency: Dict[str, int] = {}
        for result in results:
            highlights = result.get("highlights") or []
            text = " ".join(
                [result.get("title", ""), result.get("text", "")]
                + [str(h) for h in highlights if isinstance(h, str)]
            )
            for entity in _ENTITY_RE.findall(text):
                entity = entity.strip()
                if len(entity) < 3 or entity in _GENERIC_ENTITIES:
                    continue
                key = entity.lower()
                if any(key in token or token in key for token in covered):
                    continue
                frequency[entity] = frequency.get(entity, 0) + 1
        ranked = sorted(frequency, key=lambda e: (-frequency[e], e))
        return [entity for entity in ranked if frequency[entity] >= self.min_entity_freq]

    # -- Exa accessors ----------------------------------------------------

    def _search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        raw = self.exa.search(query=query, num_results=num_results)
        return self.exa.format_results(raw)

    def _find_similar(self, top_result: Dict[str, Any], num: int) -> List[Dict[str, Any]]:
        url = (top_result.get("url") or "").strip()
        if not url:
            return []
        raw = self.exa.find_similar(url=url, num_results=num)
        return self.exa.format_results(raw)

    @staticmethod
    def _mode_for_subquery(subquery: Dict[str, Any]) -> str:
        """Pick a node's search mode from the planned subquery's metadata."""
        content_type = (subquery.get("type") or "auto").lower()
        query = (subquery.get("query") or "").lower()
        if "research paper" in content_type or "pdf" in content_type or "paper" in query:
            return MODE_DEEP
        return MODE_WIDE

    @staticmethod
    def _dedupe(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop duplicate sources by normalized URL, preserving first occurrence."""
        seen: Set[str] = set()
        unique: List[Dict[str, Any]] = []
        for result in results:
            url = (result.get("url") or "").strip().lower().rstrip("/")
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            unique.append(result)
        return unique
