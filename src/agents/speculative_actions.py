"""Speculative Actions: lossless speculative tool-call middleware.

Adapted from *Speculative Actions: A Lossless Framework for Faster
Agentic Systems* (arXiv:2510.04371v2) -- Mode 2 adapted port.

The research loop is sequential: the model emits a tool call, the
supervisor executes it, the model emits the next. Each web search costs
seconds of latency. Speculative Actions overlap the *next* tool's
execution with the model's generation, the way speculative execution
overlaps a predicted branch with a CPU pipeline:

  1. ``prime``   -- after a tool runs, a draft predicts the next action
                    and pre-executes it on a background thread.
  2. ``execute`` -- when the model emits its real action, compare it
                    exactly; on a match, serve the pre-executed result
                    (a *hit*); on a miss, execute the real action fresh.

Verify-on-match is what makes the framework **lossless**: the model is
never handed a result for an action it did not choose. For read-only,
idempotent tools (Exa search) the served result is identical to a fresh
execution; for tools whose result is an LLM synthesis the *action
trajectory* is still preserved -- the model always sees a result for
exactly the ``(tool, input)`` it picked, just computed one step earlier.

Mode 2 substitution: the paper's learned draft model is replaced by a
parameter-free :class:`WorkflowPredictor` (a deterministic state machine
over this pipeline's canonical workflow), and its general concurrency
runtime by a single in-flight daemon thread.
"""

import json
import threading
from typing import Any, Callable, Dict, Optional, Tuple

# A tool executor honours the supervisor's (tool_name, tool_input) -> result
# contract used throughout the research loop.
ToolExecutor = Callable[[str, Dict[str, Any]], str]
# A draft prediction: the next (tool_name, tool_input) to pre-execute, or None
# when the draft declines to predict.
PredictedAction = Optional[Tuple[str, Dict[str, Any]]]


class WorkflowPredictor:
    """Parameter-free draft model for the speculate step.

    The paper's draft is a small LLM that guesses the next action from the
    conversation. We replace it with a deterministic state machine over this
    pipeline's canonical workflow: the planning agent is always followed by
    the retriever, which consumes the planner's subquery blob verbatim under
    the same research query. That makes the prediction exact-input, so the
    verify step can hit -- and the retriever (Exa search plus synthesis) is
    the most expensive tool in the loop, so a hit elides its full latency.
    """

    PLANNING_TOOL = "planning_agent"
    RETRIEVER_TOOL = "web_search_retriever"

    def predict_next(
        self,
        last_tool: str,
        last_input: Dict[str, Any],
        last_result: str,
    ) -> PredictedAction:
        """Return the speculated next action, or None to decline."""
        if last_tool == self.PLANNING_TOOL:
            return (
                self.RETRIEVER_TOOL,
                {
                    "research_query": last_input.get("research_query", ""),
                    # The planner's output is exactly the subqueries blob the
                    # retriever expects as subqueries_json.
                    "subqueries_json": last_result,
                },
            )
        return None


class _Pending:
    """A single in-flight speculative execution."""

    __slots__ = ("tool", "key", "result", "error", "thread")

    def __init__(self, tool: str, key: str) -> None:
        self.tool = tool
        self.key = key
        self.result: Optional[str] = None
        self.error: Optional[Exception] = None
        self.thread: Optional[threading.Thread] = None


class SpeculativeActions:
    """Lossless speculative-execution middleware for the tool loop.

    Wraps the supervisor's ``(tool_name, tool_input) -> result`` executor so
    the predicted next call pre-executes on a background thread and is served
    on an exact match. Off by default; the supervisor opts in via
    ``enable_speculative_actions()``.
    """

    def __init__(
        self,
        executor: ToolExecutor,
        predictor: Optional[WorkflowPredictor] = None,
    ) -> None:
        self._executor = executor
        self._predictor = predictor or WorkflowPredictor()
        self._pending: Optional[_Pending] = None
        self._lock = threading.Lock()
        # Accounting surfaced for diagnostics / reports.
        self.predictions = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _input_key(tool_input: Dict[str, Any]) -> str:
        """Stable, comparable fingerprint of a tool input dict."""
        return json.dumps(tool_input, sort_keys=True, default=str)

    def prime(
        self,
        last_tool: str,
        last_input: Dict[str, Any],
        last_result: str,
    ) -> None:
        """Speculatively pre-execute the predicted next action.

        No-op when the draft declines to predict. Safe to call from the
        supervisor's tool loop right after a tool returns; the work runs on a
        daemon thread so it overlaps the model's next generation step and
        never blocks process exit.
        """
        predicted = self._predictor.predict_next(last_tool, last_input, last_result)
        if predicted is None:
            return
        tool, tool_input = predicted
        pending = _Pending(tool, self._input_key(tool_input))

        def _run() -> None:
            try:
                pending.result = self._executor(tool, tool_input)
            except Exception as exc:  # speculation failure -> treated as a miss
                pending.error = exc

        thread = threading.Thread(target=_run, name="speculative-action", daemon=True)
        pending.thread = thread
        with self._lock:
            # A previous speculation we never matched is dropped; its result is
            # simply discarded (lossless, but wasted work -- the paper's
            # accepted cost of speculation).
            self._pending = pending
        self.predictions += 1
        thread.start()

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Verify-on-match: serve the speculation on an exact hit, else run fresh.

        A call with no outstanding speculation is served by a plain execution
        and counts as neither hit nor miss. A *miss* is a speculation that was
        outstanding but did not match (the model chose a different action) or
        that raised -- i.e. wasted pre-execution.
        """
        key = self._input_key(tool_input)
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is not None:
            if (
                pending.thread is not None
                and pending.tool == tool_name
                and pending.key == key
            ):
                pending.thread.join()  # overlap ends here; side effects visible
                if pending.error is None and pending.result is not None:
                    self.hits += 1
                    return pending.result
            # Speculation was outstanding but missed (or failed): the
            # pre-execution is discarded and the real action runs fresh.
            self.misses += 1
        return self._executor(tool_name, tool_input)

    @property
    def stats(self) -> Dict[str, Any]:
        served = self.hits + self.misses
        return {
            "predictions": self.predictions,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / served) if served else 0.0,
        }
