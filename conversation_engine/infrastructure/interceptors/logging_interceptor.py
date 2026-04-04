"""
LoggingInterceptor — Logs node entry, exit, and errors.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from conversation_engine.infrastructure.instrumented_graph import Interceptor

logger = logging.getLogger(__name__)


class LoggingInterceptor(Interceptor):
    """
    Logs before/after/error for every node invocation.

    Stores per-call start time in a dict keyed by node_name.
    For thread safety with parallel Send() fan-outs, consider
    keying by (node_name, thread_id) — left simple for now.
    """

    def __init__(self, level: int = logging.INFO):
        self._level = level
        self._start_times: dict[str, float] = {}

    def before(self, node_name: str, state: Any) -> None:
        self._start_times[node_name] = time.perf_counter()
        logger.log(self._level, "[%s] ▶ entering", node_name)

    def after(self, node_name: str, state: Any, result: Any) -> None:
        elapsed = time.perf_counter() - self._start_times.pop(node_name, 0.0)
        logger.log(self._level, "[%s] ✓ completed (%.3fs)", node_name, elapsed)

    def on_error(self, node_name: str, state: Any, error: Exception) -> None:
        elapsed = time.perf_counter() - self._start_times.pop(node_name, 0.0)
        logger.error("[%s] ✗ failed after %.3fs: %s", node_name, elapsed, error)
