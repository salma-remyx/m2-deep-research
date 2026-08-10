"""Analytic memory over gathered research sources.

Adapted from **AdaMM** -- *Beyond Retrieval: Analytic Memory for Multimodal
Agents* (arXiv:2607.29440v1). AdaMM formulates *analytic memory* as a
complementary abstraction to retrieval memory: rather than only returning
query-relevant records, it organizes recurring observations into queryable
structures that support **filtering, aggregation, ranking, and temporal
comparison** -- i.e. *computing* over accumulated evidence. At inference time
a memory-aware planner decomposes a query into retrieval and analytic
operations and routes each to the right tool.

This is a **Mode 2 (adapted port)** of that mechanism for this deep-research
pipeline:

* AdaMM extracts provenance-linked attribute-value observations from
  multimodal *dialogue and images* via learned extractors. We replace those
  with a **parameter-free extractor over the gathered Exa source records**
  (numeric measurements with their unit and source provenance). No learned
  components, no extra API keys, fully deterministic.
* AdaMM's memory-aware planner decomposes a query and routes ops to
  specialized tools via a learned policy. We replace the learned planner with
  a **deterministic analytic tool** the supervisor registers and calls with a
  single structured op; the supervisor (MiniMax-M2.1) plays the planner role,
  choosing which analytic op to run during synthesis.
* AdaMM's MemEye / MemGallery multimodal benchmarks are **not ported** --
  evaluation belongs in a downstream PR. This module is the analytic-memory
  primitive plus its supervisor wiring.

The core mechanism is preserved: recurring attribute-value observations are
extracted from gathered evidence, recurring field structures are discovered
and materialized, and the supervisor can compute over them (filter,
aggregate, rank, compare over time) instead of re-reading raw search output.

**Scope / honesty note.** The discovered ``field`` is a parameter-free
context-window proxy (the significant tokens immediately preceding a numeric
mention), not a learned attribute label; recurrence is detected on that
proxy. This trades AdaMM's schema-discovery accuracy for determinism and zero
extra cost, which is the intended Mode 2 trade.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# A numeric mention: digit groups with optional thousands separators /
# decimals, or a small written-out number. Used both to find mentions and to
# read the surrounding context window for the field label.
_NUM_RE = re.compile(
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|hundred)\b)"
)

# Written-out numbers -> numeric value, for prose like "below one percent".
_WORD_NUM: Dict[str, float] = {
    "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0,
    "fifteen": 15.0, "twenty": 20.0, "thirty": 30.0, "forty": 40.0,
    "fifty": 50.0, "sixty": 60.0, "seventy": 70.0, "eighty": 80.0,
    "ninety": 90.0, "hundred": 100.0,
}

# Scale words that magnify the preceding number.
_SCALE: Dict[str, float] = {
    "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
}

# Dimension words that name what a number measures; first match wins.
_UNIT: Dict[str, str] = {
    "%": "percent", "percent": "percent", "percentage": "percent",
    "$": "dollars", "dollar": "dollars", "dollars": "dollars", "usd": "dollars",
    "gb": "gb", "tb": "tb", "mb": "mb", "pflops": "pflops", "tflops": "tflops",
    "km": "km", "kg": "kg", "tons": "tons", "tonnes": "tons",
    "qubits": "qubits", "nodes": "nodes", "users": "users",
}

# Tokens of >= 3 lowercase alphanumeric chars (drops punctuation/short noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Common words ignored when labeling a field, so surface words carry the label.
_STOPWORDS: frozenset = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over
    under about as is are was were be been being this that these those it its
    their our your his her they them we you he she him not no nor so than too
    very can could should would may might must will shall do does did has have
    had more most less least many much few several also however which who whom
    whose what when where why how during while across among between within
    without via per using used use new one two three first second next last
    according based recent currently reportedly said says estimated projected
    expected predicted reached reach nearly around approximately up down out
    now still since than some any each both either neither
    """.split()
)

# Tokens that qualify a number (scale / unit words) rather than naming the
# attribute it measures -- excluded from the field label so they do not leak
# into the discovered attribute name.
_QUALIFIER_TOKENS: frozenset = _STOPWORDS | set(_SCALE) | set(_UNIT) | set(
    _UNIT.values()
)

_OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

_VALID_FUNCS = ("count", "sum", "avg", "min", "max")


@dataclass
class Observation:
    """A single provenance-linked attribute-value observation."""

    field: str
    value: float
    unit: str
    text: str
    source_url: str = ""
    source_title: str = ""
    published_date: Optional[str] = None
    subquery: str = ""

    def as_row(self) -> Dict[str, Any]:
        """Flat dict view for analytic output."""
        return {
            "field": self.field,
            "value": self.value,
            "unit": self.unit,
            "source": self.source_url,
            "title": self.source_title,
            "date": self.published_date,
            "text": self.text,
        }


@dataclass
class AnalyticResult:
    """Outcome of an analytic-memory operation, renderable as markdown."""

    operation: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    schema: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def render(self) -> str:
        """Render as a concise markdown block for the supervisor to consume."""
        lines: List[str] = ["", f"### Analytic memory — `{self.operation}`", ""]
        if self.note:
            lines.append(f"_{self.note}_")
            lines.append("")
        if self.schema:
            lines.append("**Discovered fields:**")
            for f in self.schema:
                rec = " (recurring)" if f.get("recurring") else ""
                lines.append(
                    f"- `{f['field']}` — {f['observations']} obs / "
                    f"{f['sources']} source(s){rec} "
                    f"[range {f['min']:.4g}..{f['max']:.4g} {f['units']}]"
                )
            lines.append("")
        if self.summary is not None:
            lines.append("**Result:**")
            for key, val in self.summary.items():
                lines.append(f"- {key}: {val}")
            lines.append("")
        if self.rows:
            lines.append("**Observations:**")
            for row in self.rows:
                lines.append(
                    f"- {row['value']:g} {row['unit']} — `{row['field']}` "
                    f"({row.get('title') or row.get('source') or 'n/a'})"
                )
            lines.append("")
        if not self.schema and not self.rows and self.summary is None:
            lines.append("_No matching observations in analytic memory._")
            lines.append("")
        return "\n".join(lines)


class AnalyticMemory:
    """Compute over gathered evidence beyond flat retrieval.

    Ingest the retriever's gathered sources once (after a
    ``web_search_retriever`` call), then answer analytic queries
    (filter / aggregate / rank / temporal series) over the extracted
    observations. Each observation carries its source provenance so every
    computed number traces back to a retrieved record.
    """

    def __init__(self, field_window: int = 3, min_field_observations: int = 2):
        self.field_window = field_window
        self.min_field_observations = min_field_observations
        self._observations: List[Observation] = []

    # -- ingestion ---------------------------------------------------------

    def ingest(self, sources: Iterable[Any]) -> int:
        """Extract observations from gathered sources.

        Accepts either a flat list of source dicts (``url`` / ``title`` /
        ``text`` / ``highlights`` / ``published_date``) or the nested
        subquery buckets emitted by ``WebSearchRetriever``.

        Returns the number of observations added.
        """
        before = len(self._observations)
        for src in self._iter_sources(sources):
            self._observations.extend(self._extract(src))
        return len(self._observations) - before

    def reset(self) -> None:
        """Clear all observations."""
        self._observations = []

    @property
    def observations(self) -> List[Observation]:
        return list(self._observations)

    # -- analytic surface --------------------------------------------------

    def discovered_fields(self) -> List[Dict[str, Any]]:
        """Materialize recurring field structures from the observation set."""
        groups: Dict[str, List[Observation]] = defaultdict(list)
        for obs in self._observations:
            groups[obs.field].append(obs)
        schema: List[Dict[str, Any]] = []
        for name, group in groups.items():
            values = [o.value for o in group]
            units = sorted({o.unit for o in group})
            distinct_sources = {o.source_url or o.source_title for o in group}
            schema.append(
                {
                    "field": name,
                    "observations": len(group),
                    "sources": len(distinct_sources),
                    "recurring": len(group) >= self.min_field_observations,
                    "min": min(values),
                    "max": max(values),
                    "units": ", ".join(units),
                }
            )
        schema.sort(key=lambda f: (-f["observations"], f["field"]))
        return schema

    def query(self, operation: str, **params: Any) -> AnalyticResult:
        """Dispatch a single analytic operation.

        Supported operations: ``summarize``, ``filter``, ``aggregate``,
        ``rank``, ``series``.
        """
        op = (operation or "").strip().lower()
        if op == "summarize":
            return self.summarize()
        if op == "filter":
            return self._filter(**params)
        if op == "aggregate":
            return self._aggregate(**params)
        if op == "rank":
            return self._rank(**params)
        if op == "series":
            return self._series(**params)
        return AnalyticResult(
            operation=op,
            note=(
                f"Unknown operation '{operation}'. Supported: "
                "summarize, filter, aggregate, rank, series."
            ),
        )

    def summarize(self) -> AnalyticResult:
        """Return the discovered schema plus a sample of observations."""
        schema = self.discovered_fields()
        sample = [o.as_row() for o in self._observations[:8]]
        return AnalyticResult(
            operation="summarize",
            rows=sample,
            schema=schema,
            summary={
                "observations": len(self._observations),
                "fields": len(schema),
                "recurring_fields": sum(1 for f in schema if f["recurring"]),
            },
        )

    # -- operations --------------------------------------------------------

    def _select(self, field: Optional[str], unit: Optional[str]) -> List[Observation]:
        fld = (field or "").strip().lower() or None
        un = (unit or "").strip().lower() or None
        return [
            o for o in self._observations
            if (fld is None or o.field == fld) and (un is None or o.unit == un)
        ]

    def _filter(
        self,
        field: Optional[str] = None,
        unit: Optional[str] = None,
        op: Optional[str] = None,
        value: Optional[float] = None,
        **_: Any,
    ) -> AnalyticResult:
        selected = self._select(field, unit)
        if op:
            cmp = _OPS.get(op)
            if cmp is None:
                return AnalyticResult(
                    operation="filter",
                    note=f"Unknown comparator '{op}'. Supported: {sorted(_OPS)}.",
                )
            if value is None:
                return AnalyticResult(
                    operation="filter", note="Comparator 'op' requires 'value'."
                )
            selected = [o for o in selected if cmp(o.value, value)]
        return AnalyticResult(
            operation="filter", rows=[o.as_row() for o in selected]
        )

    def _aggregate(
        self,
        func: str = "count",
        field: Optional[str] = None,
        unit: Optional[str] = None,
        group_by: Optional[str] = None,
        **_: Any,
    ) -> AnalyticResult:
        fn = (func or "count").strip().lower()
        if fn not in _VALID_FUNCS:
            return AnalyticResult(
                operation="aggregate",
                note=f"Unknown func '{func}'. Supported: {list(_VALID_FUNCS)}.",
            )
        selected = self._select(field, unit)
        summary: Dict[str, Any] = {}

        def _stat(obs: List[Observation]) -> Dict[str, Any]:
            values = [o.value for o in obs]
            if not values:
                return {"count": 0}
            if fn == "count":
                return {"count": len(values)}
            if fn == "sum":
                return {"sum": round(sum(values), 6)}
            if fn == "avg":
                return {"avg": round(sum(values) / len(values), 6)}
            if fn == "min":
                return {"min": min(values)}
            return {"max": max(values)}

        if group_by:
            key_attr = "unit" if group_by == "unit" else "field"
            groups: Dict[str, List[Observation]] = defaultdict(list)
            for o in selected:
                groups[getattr(o, key_attr)].append(o)
            for key in sorted(groups):
                summary[str(key)] = _stat(groups[key])
        else:
            summary = _stat(selected)
        return AnalyticResult(operation="aggregate", summary=summary)

    def _rank(
        self,
        field: Optional[str] = None,
        unit: Optional[str] = None,
        n: int = 5,
        order: str = "desc",
        **_: Any,
    ) -> AnalyticResult:
        selected = sorted(
            self._select(field, unit), key=lambda o: o.value,
            reverse=(order != "asc"),
        )
        try:
            limit = max(0, int(n))
        except (TypeError, ValueError):
            limit = 5
        return AnalyticResult(
            operation="rank", rows=[o.as_row() for o in selected[:limit]]
        )

    def _series(
        self, field: Optional[str] = None, unit: Optional[str] = None, **_: Any
    ) -> AnalyticResult:
        """Temporal comparison: observations ordered by publication date."""
        selected = self._select(field, unit)

        def _year(o: Observation) -> Tuple[int, str]:
            return (_parse_year(o.published_date), o.published_date or "")

        ordered = sorted(selected, key=_year)
        return AnalyticResult(
            operation="series", rows=[o.as_row() for o in ordered]
        )

    # -- extraction internals ---------------------------------------------

    def _iter_sources(self, sources: Iterable[Any]) -> Iterator[Dict[str, Any]]:
        """Yield flat source dicts, flattening retriever subquery buckets."""
        for src in sources or []:
            if not isinstance(src, dict):
                continue
            if "results" in src or "similar_results" in src:
                subq = src.get("subquery", "")
                for bucket_key in ("results", "similar_results"):
                    for item in src.get(bucket_key) or []:
                        if isinstance(item, dict):
                            flat = dict(item)
                            flat.setdefault("subquery", subq)
                            yield flat
            else:
                yield src

    def _extract(self, src: Dict[str, Any]) -> List[Observation]:
        """Pull provenance-linked numeric observations out of one source."""
        url = src.get("url", "") or ""
        title = src.get("title", "") or ""
        date = src.get("published_date") or src.get("publishedDate")
        subq = src.get("subquery", "") or ""
        parts = [title, src.get("text", "") or ""]
        highlights = src.get("highlights")
        if isinstance(highlights, list):
            parts.append(" ".join(str(h) for h in highlights))
        # Keep a single blob for field-label context, but record the exact
        # matched span as the observation's provenance text.
        blob = "\n".join(p for p in parts if p)
        out: List[Observation] = []
        # Dedup within a source by (value, unit): the same fact restated in the
        # title, body, and highlights is one observation, not three.
        seen_values: set = set()
        for match in _NUM_RE.finditer(blob):
            value, unit, end = self._read_number(blob, match)
            if value is None:
                continue
            key = (value, unit)
            if key in seen_values:
                continue
            seen_values.add(key)
            field_label = self._field_label(blob[: match.start()])
            span = blob[match.start(): min(len(blob), end + 24)].strip()
            out.append(
                Observation(
                    field=field_label,
                    value=value,
                    unit=unit,
                    text=span[:160],
                    source_url=url,
                    source_title=title,
                    published_date=date,
                    subquery=subq,
                )
            )
        return out

    def _read_number(
        self, text: str, match: "re.Match[str]"
    ) -> Tuple[Optional[float], str, int]:
        """Parse value + unit from a numeric match; return (value, unit, end).

        Returns ``value=None`` to signal "skip" -- used for bare 4-digit years
        (1900-2099), which are temporal markers rather than measurements; the
        source's ``published_date`` already carries temporality for series ops.
        """
        raw = match.group("num")
        raw_lower = raw.lower()
        value = _WORD_NUM[raw_lower] if raw_lower in _WORD_NUM else _to_float(raw)
        if value is None:
            return None, "count", match.end()
        tail = text[match.end():].lstrip()
        # Greedy scale word first ("47 billion"), then unit ("dollars").
        scaled = False
        rest = tail
        first_word = _first_word(rest)
        if first_word in _SCALE:
            value *= _SCALE[first_word]
            scaled = True
            rest = rest[len(first_word):].lstrip()
            first_word = _first_word(rest)
        if first_word in _UNIT:
            unit = _UNIT[first_word]
        else:
            unit = "count"
        # A bare 4-digit token with no scale and no unit is a year, not a
        # measurement -- drop it so it does not seed a false recurring field.
        if (
            unit == "count"
            and not scaled
            and raw.isdigit()
            and len(raw) == 4
            and 1900 <= value <= 2099
        ):
            return None, "year", match.end()
        return value, unit, match.end()

    def _field_label(self, preceding: str) -> str:
        """Context-window proxy for the attribute a number measures."""
        tokens = [
            t
            for t in _TOKEN_RE.findall(preceding.lower())
            if t not in _QUALIFIER_TOKENS and not t.isdigit()
        ]
        if not tokens:
            return "_numeric"
        window = tokens[-self.field_window:]
        return "_".join(window)


def _to_float(raw: str) -> Optional[float]:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _first_word(text: str) -> str:
    m = re.match(r"[A-Za-z%$]+", text)
    return m.group(0).lower() if m else ""


def _parse_year(date_str: Optional[str]) -> int:
    """Best-effort year extraction from an Exa published_date (ISO-ish)."""
    if not date_str:
        return 9999  # undated observations sort last in temporal series
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(date_str))
    return int(m.group(1)) if m else 9999
