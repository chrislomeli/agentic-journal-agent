"""
middleware — Composable cross-cutting concerns for LangGraph nodes.

One ABC, one pattern: each middleware wraps the next, forming a call chain.
The innermost call is the actual node function.

    Logging → Metrics → Validation → Retry → CircuitBreaker → [node]

Each middleware receives (node_name, state, next_fn) and decides:
  - Call next_fn(state) to continue the chain
  - Don't call it — to short-circuit (circuit breaker)
  - Call it multiple times — to retry
  - Transform state before or result after

Per-node selectivity: each middleware carries an optional `nodes` set.
If provided, the middleware only activates for those nodes; otherwise it
applies to all nodes.

This replaces the former Interceptor (observe-only) and Middleware
(transform-only) ABCs with a single, composable pattern.
"""

from conversation_engine.infrastructure.middleware.base import (
    NodeMiddleware,
)
from conversation_engine.infrastructure.middleware.logging_mw import (
    LoggingMiddleware,
)
from conversation_engine.infrastructure.middleware.metrics_mw import (
    MetricsMiddleware,
    NodeMetrics,
)
from conversation_engine.infrastructure.middleware.validation_mw import (
    ValidationMiddleware,
)
from conversation_engine.infrastructure.middleware.error_handling_mw import (
    ErrorHandlingMiddleware,
)
from conversation_engine.infrastructure.middleware.retry_mw import (
    RetryMiddleware,
)
from conversation_engine.infrastructure.middleware.circuit_breaker_mw import (
    CircuitBreakerMiddleware,
)
from conversation_engine.infrastructure.middleware.config_mw import (
    ConfigMiddleware,
)

__all__ = [
    "NodeMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "NodeMetrics",
    "ValidationMiddleware",
    "ErrorHandlingMiddleware",
    "RetryMiddleware",
    "CircuitBreakerMiddleware",
    "ConfigMiddleware",
]
