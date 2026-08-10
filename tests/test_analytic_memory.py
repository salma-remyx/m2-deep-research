"""Tests for the analytic memory and its supervisor wiring.

Analytic memory is adapted from AdaMM -- *Beyond Retrieval: Analytic Memory
for Multimodal Agents* (arXiv:2607.29440v1). These tests cover both the
:class:`~src.agents.analytic_memory.AnalyticMemory` in isolation and its
integration into the existing :class:`~src.agents.supervisor.SupervisorAgent`
-- the call site -- which is what proves the wiring actually invokes the new
code (the retriever feeds the memory, and the supervisor's tool dispatch
serves analytic queries).
"""

import pytest

from src.agents.analytic_memory import AnalyticMemory, AnalyticResult, Observation
from src.agents.supervisor import SupervisorAgent  # non-new module -> proves wiring

# Nested buckets mirroring WebSearchRetriever.search_with_subqueries output;
# each leaf is a flat source dict from ExaTool.format_results.
ANALYTIC_SOURCES = [
    {
        "subquery": "quantum market size",
        "priority": 1,
        "results": [
            {
                "title": "Quantum Computing Market Report",
                "url": "https://example.com/quantum-report",
                "published_date": "2025-03-01T00:00:00.000Z",
                "text": "The quantum computing market is projected to reach "
                "47 billion dollars.",
                "highlights": [],
            },
        ],
        "similar_results": [
            {
                "title": "Global Quantum Outlook",
                "url": "https://example.com/quantum-outlook",
                "published_date": "2024-06-15T00:00:00.000Z",
                "text": "The quantum market reached 9 billion dollars.",
                "highlights": [],
            },
        ],
    },
    {
        "subquery": "qubit fidelity",
        "priority": 2,
        "results": [
            {
                "title": "Qubit Fidelity Research",
                "url": "https://example.com/qubit-fidelity",
                "published_date": "2025-01-10T00:00:00.000Z",
                "text": "Error rates in superconducting qubits dropped below "
                "1 percent over the last year.",
                "highlights": ["error rates below 1 percent"],
            },
        ],
        "similar_results": [],
    },
]

# Two sources whose numeric mentions share the same preceding context, so the
# discovered field recurs across distinct sources.
THROUGHPUT_SOURCES = [
    {
        "title": "Latency Benchmarks A",
        "url": "https://example.com/bench-a",
        "published_date": "2025-01-01T00:00:00.000Z",
        "text": "Benchmark throughput measured 5000 requests per run.",
        "highlights": [],
    },
    {
        "title": "Latency Benchmarks B",
        "url": "https://example.com/bench-b",
        "published_date": "2025-02-01T00:00:00.000Z",
        "text": "Benchmark throughput measured 9000 requests per run.",
        "highlights": [],
    },
]


@pytest.fixture
def patch_config_keys(monkeypatch):
    """Let SupervisorAgent() construct without real API credentials."""
    from src.utils.config import Config

    monkeypatch.setattr(Config, "MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EXA_API_KEY", "test-key")


# --------------------------------------------------------------------------- #
# AnalyticMemory unit tests (new module)
# --------------------------------------------------------------------------- #


def test_ingest_extracts_scaled_currency_and_percent():
    mem = AnalyticMemory()
    added = mem.ingest(ANALYTIC_SOURCES)

    assert added == len(mem.observations)
    dollars = sorted(o.value for o in mem.observations if o.unit == "dollars")
    # 47 billion and 9 billion dollars, scaled into the value.
    assert dollars == [9_000_000_000.0, 47_000_000_000.0]
    percents = [o.value for o in mem.observations if o.unit == "percent"]
    assert percents == [1.0]


def test_ingest_carries_provenance():
    mem = AnalyticMemory()
    mem.ingest(ANALYTIC_SOURCES)

    obs = next(o for o in mem.observations if o.unit == "percent")
    assert obs.source_url == "https://example.com/qubit-fidelity"
    assert obs.source_title == "Qubit Fidelity Research"
    assert obs.subquery == "qubit fidelity"


def test_ingest_skips_bare_years_and_dedups_within_source():
    mem = AnalyticMemory()
    src = [
        {
            "title": "t",
            "url": "https://example.com/x",
            "published_date": "2025-01-01T00:00:00.000Z",
            # Same fact in text + highlight (dedups to one), plus a bare year
            # (2024) that must NOT become a measurement observation.
            "text": "Revenue hit 5 million dollars in 2024.",
            "highlights": ["revenue hit 5 million dollars"],
        }
    ]
    mem.ingest(src)

    values = [(o.value, o.unit) for o in mem.observations]
    assert values == [(5_000_000.0, "dollars")]
    assert not any(o.unit == "year" for o in mem.observations)


def test_discovered_fields_marks_recurring_fields():
    mem = AnalyticMemory()
    mem.ingest(THROUGHPUT_SOURCES)

    schema = {f["field"]: f for f in mem.discovered_fields()}
    # Same preceding context -> same field slug -> recurring across 2 sources.
    recurring = [name for name, f in schema.items() if f["recurring"]]
    assert recurring, f"expected a recurring field, got {schema}"
    field = recurring[0]
    assert schema[field]["sources"] == 2
    assert schema[field]["observations"] == 2
    assert schema[field]["min"] == 5000.0
    assert schema[field]["max"] == 9000.0


def test_aggregate_avg_and_group_by_unit():
    mem = AnalyticMemory()
    mem.ingest(ANALYTIC_SOURCES)

    result = mem.query("aggregate", func="avg", group_by="unit")
    assert isinstance(result, AnalyticResult)
    # Mean of 47bn and 9bn dollars is 28bn.
    assert result.summary["dollars"]["avg"] == pytest.approx(28_000_000_000.0)
    assert result.summary["percent"]["avg"] == pytest.approx(1.0)


def test_aggregate_count_without_grouping():
    mem = AnalyticMemory()
    mem.ingest(ANALYTIC_SOURCES)

    result = mem.query("aggregate", func="count", unit="dollars")
    assert result.summary == {"count": 2}


def test_aggregate_rejects_unknown_func():
    mem = AnalyticMemory()
    result = mem.query("aggregate", func="median")
    assert "median" in result.note


def test_filter_by_value_comparator():
    mem = AnalyticMemory()
    mem.ingest(ANALYTIC_SOURCES)

    result = mem.query("filter", unit="dollars", op=">", value=10_000_000_000.0)
    values = [r["value"] for r in result.rows]
    assert values == [47_000_000_000.0]


def test_rank_orders_descending_by_default():
    mem = AnalyticMemory()
    mem.ingest(ANALYTIC_SOURCES)

    result = mem.query("rank", n=2)
    values = [r["value"] for r in result.rows]
    assert values[0] > values[-1]
    assert values[0] == 47_000_000_000.0


def test_series_orders_by_publication_date():
    mem = AnalyticMemory()
    mem.ingest(ANALYTIC_SOURCES)

    result = mem.query("series", unit="dollars")
    dates = [r["date"] for r in result.rows]
    # Earlier published_date (2024-06) sorts before later (2025-03).
    assert dates == sorted(dates)
    assert result.rows[0]["title"] == "Global Quantum Outlook"


def test_summarize_reports_schema_and_counts():
    mem = AnalyticMemory()
    mem.ingest(ANALYTIC_SOURCES)

    result = mem.summarize()
    rendered = result.render()
    assert "Discovered fields" in rendered
    assert result.summary["observations"] == len(mem.observations)
    assert result.summary["fields"] >= 1


def test_query_unknown_operation_is_reported_not_raised():
    mem = AnalyticMemory()
    result = mem.query("frobnicate")
    assert "summarize" in result.note and "filter" in result.note


def test_empty_sources_yield_empty_memory():
    mem = AnalyticMemory()
    mem.ingest([])
    assert mem.observations == []
    assert mem.discovered_fields() == []
    assert mem.summarize().summary["observations"] == 0


def test_accepts_flat_source_list():
    mem = AnalyticMemory()
    mem.ingest(THROUGHPUT_SOURCES)  # already flat dicts
    assert len(mem.observations) == 2


# --------------------------------------------------------------------------- #
# Integration: exercises the wiring inside the existing SupervisorAgent
# --------------------------------------------------------------------------- #


def test_supervisor_instantiates_analytic_memory(patch_config_keys):
    supervisor = SupervisorAgent()

    assert isinstance(supervisor.analytic_memory, AnalyticMemory)
    assert supervisor.analytic_memory.observations == []
    assert "analytic_memory" in {t["name"] for t in supervisor.tools}


def test_supervisor_retriever_feeds_analytic_memory(patch_config_keys):
    supervisor = SupervisorAgent()
    # Stub the retriever so no network call is made.
    supervisor.web_search_retriever.retrieve = lambda q, s: "findings"
    supervisor.web_search_retriever.last_search_results = ANALYTIC_SOURCES

    supervisor.execute_tool(
        "web_search_retriever",
        {"research_query": "q", "subqueries_json": '{"subqueries": []}'},
    )

    # Ingest happened automatically as part of the retriever tool call.
    assert supervisor._gathered_sources == ANALYTIC_SOURCES
    assert len(supervisor.analytic_memory.observations) > 0
    dollars = [o.value for o in supervisor.analytic_memory.observations if o.unit == "dollars"]
    assert 47_000_000_000.0 in dollars


def test_supervisor_analytic_memory_tool_dispatches_query(patch_config_keys):
    supervisor = SupervisorAgent()
    supervisor._gathered_sources = ANALYTIC_SOURCES
    supervisor.analytic_memory.ingest(ANALYTIC_SOURCES)

    out = supervisor.execute_tool(
        "analytic_memory",
        {"operation": "aggregate", "func": "avg", "group_by": "unit"},
    )

    assert "Analytic memory" in out
    # The computed mean (28bn dollars) is surfaced to the supervisor as text.
    assert "28000000000.0" in out


def test_supervisor_analytic_memory_tool_handles_unknown_op(patch_config_keys):
    supervisor = SupervisorAgent()

    out = supervisor.execute_tool("analytic_memory", {"operation": "nope"})

    assert "Unknown operation" in out


def test_observation_as_row_round_trips_fields():
    obs = Observation(
        field="market", value=47.0, unit="dollars", text="reach 47 billion",
        source_url="https://example.com/x", source_title="t",
        published_date="2025-01-01", subquery="size",
    )
    row = obs.as_row()
    assert row["field"] == "market"
    assert row["value"] == 47.0
    assert row["source"] == "https://example.com/x"
