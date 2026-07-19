"""Integration tests for the verifier-backed task-state wiring.

These exercise the call site (``SupervisorAgent.research()`` in the
NON-NEW module ``src.agents.supervisor``) with a mocked Anthropic-style
streaming client, so no network or API keys are required. They assert
the integrated StructAgent behavior: verifier-gated commits, attributed
failures, and a state-grounded exhaustion report.
"""

import os
import pathlib
import sys
from types import SimpleNamespace

# Dummy API keys must be set BEFORE importing ``src`` (Config reads them at import).
os.environ.setdefault("MINIMAX_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.supervisor import SupervisorAgent  # noqa: E402
from src.agents.task_state import Transition  # noqa: E402


def _tool_use_block(name, input_, idx):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=f"tu_{name}_{idx}")


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


class _FakeStream:
    """Mimics the Anthropic streaming context manager used by ``research()``."""

    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())  # no streaming events; the for-loop is a no-op

    def get_final_message(self):
        return self._message


class _FakeMessages:
    """Cycles through a scripted list of messages, one per ``stream()`` call."""

    def __init__(self, script):
        self._script = script
        self._i = 0

    def stream(self, **kwargs):
        message = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return _FakeStream(message)


def _build_supervisor(execute_tool):
    """Construct a SupervisorAgent with a mocked client + tool executor."""
    supervisor = SupervisorAgent()
    supervisor.client.messages = _FakeMessages([])
    supervisor.execute_tool = execute_tool
    return supervisor


def test_research_commits_verified_progress_and_attributes_failures():
    """A good plan verifies; an errored retrieval is attributed, not swallowed."""
    plan_result = (
        '{"status": "success", '
        '"subqueries": [{"query": "q1"}, {"query": "q2"}], '
        '"research_query": "topic"}'
    )
    retrieval_error = "Error in retrieval: connection failed"

    def fake_execute(tool_name, tool_input):
        if tool_name == "planning_agent":
            return plan_result
        if tool_name == "web_search_retriever":
            return retrieval_error
        return "Error: unknown tool"

    supervisor = _build_supervisor(fake_execute)
    # Iteration 1: model requests both tools. Iteration 2: model ends turn.
    supervisor.client.messages = _FakeMessages(
        [
            SimpleNamespace(
                content=[
                    _tool_use_block("planning_agent", {"research_query": "topic"}, 0),
                    _tool_use_block("web_search_retriever", {"research_query": "topic"}, 0),
                ],
                stop_reason="tool_use",
            ),
            SimpleNamespace(content=[_text_block("FINAL REPORT")], stop_reason="end_turn"),
        ]
    )

    report = supervisor.research("topic", max_iterations=5)

    # Happy path: final text is returned unchanged.
    assert report == "FINAL REPORT"
    # Verifier-gated: plan committed, retrieval attributed as a failure.
    assert supervisor.task_state.total == 2
    assert supervisor.task_state.committed == 1
    failures = supervisor.task_state.failures
    assert len(failures) == 1
    assert failures[0].tool == "web_search_retriever"
    assert "error" in failures[0].reason.lower()
    # Not done: no verified retrieval evidence.
    assert supervisor.task_state.is_done() is False


def test_exhaustion_returns_grounded_report_when_done():
    """At max_iterations the outcome is state-grounded, not a blind string."""
    plan_result = (
        '{"status": "success", '
        '"subqueries": [{"query": "q1"}], '
        '"research_query": "topic"}'
    )
    # Retrieval with real source URLs and enough body to verify.
    retrieval_ok = (
        "Findings on the topic: primary analysis at "
        "https://example.com/a and supporting data at "
        "https://example.com/b. " + "detail " * 40
    )

    def fake_execute(tool_name, tool_input):
        if tool_name == "planning_agent":
            return plan_result
        if tool_name == "web_search_retriever":
            return retrieval_ok
        return "Error: unknown tool"

    supervisor = _build_supervisor(fake_execute)
    # Every iteration requests both tools; the loop is driven to exhaustion.
    supervisor.client.messages = _FakeMessages(
        [
            SimpleNamespace(
                content=[
                    _tool_use_block("planning_agent", {"research_query": "topic"}, 0),
                    _tool_use_block("web_search_retriever", {"research_query": "topic"}, 0),
                ],
                stop_reason="tool_use",
            )
        ]
    )

    report = supervisor.research("topic", max_iterations=2)

    # Not the old blind "reached maximum iterations without completion." string.
    assert "without completion" not in report
    # State-grounded: verified counts + DONE auditor fired.
    assert "verified tool result(s)" in report
    assert "Enough verified evidence accumulated to synthesize a report." in report
    assert supervisor.task_state.committed == 4  # 2 iterations x 2 tools
    assert supervisor.task_state.is_done() is True
    assert Transition.VERIFIED.value in report or supervisor.task_state.committed > 0
