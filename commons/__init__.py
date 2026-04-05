"""
commons — Reusable LangGraph infrastructure.

A standalone Python package with zero domain dependencies.
Drop into any LangGraph project for production-grade cross-cutting concerns.

Subsystems
----------
1. middleware     — Composable NodeMiddleware chain + InstrumentedGraph
2. tool_client   — Transport-agnostic tool contracts (MCP-ready)
3. llm           — Protocol-based LLM interaction (CallLLM + stub + OpenAI)
4. human         — Protocol-based human interaction (CallHuman + console + mock)
5. node_validation — NodeResult envelope + validated_node decorator

Dependencies: langgraph, langchain-core, langchain-openai, pydantic
"""

from commons.middleware import (
    InstrumentedGraph,
    NodeMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    NodeMetrics,
    ValidationMiddleware,
    ErrorHandlingMiddleware,
    RetryMiddleware,
    CircuitBreakerMiddleware,
    ConfigMiddleware,
)
from commons.node_validation import (
    NodeError,
    NodeResult,
    validated_node,
    handle_error,
)
from commons.tool_client import (
    ToolSpec,
    ToolRegistry,
    ToolContentBlock,
    ToolResultEnvelope,
    ToolResultMeta,
    ToolClient,
    ToolCallError,
    LocalToolClient,
    specs_to_langchain_tools,
    execute_tool_call,
)
from commons.llm import (
    CallLLM,
    LLMRequest,
    LLMResponse,
    call_llm_stub,
    make_openai_llm,
    OpenAICallLLM,
    QuizQuestion,
    LLMValidator,
    LLMValidatorReport,
    QuizResult,
    quiz_report_summary,
)
from commons.human import (
    CallHuman,
    HumanRequest,
    HumanResponse,
    ConsoleHuman,
    MockHuman,
)

__all__ = [
    # Middleware
    "InstrumentedGraph",
    "NodeMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "NodeMetrics",
    "ValidationMiddleware",
    "ErrorHandlingMiddleware",
    "RetryMiddleware",
    "CircuitBreakerMiddleware",
    "ConfigMiddleware",
    # Node validation
    "NodeError",
    "NodeResult",
    "validated_node",
    "handle_error",
    # Tool client
    "ToolSpec",
    "ToolRegistry",
    "ToolContentBlock",
    "ToolResultEnvelope",
    "ToolResultMeta",
    "ToolClient",
    "ToolCallError",
    "LocalToolClient",
    "specs_to_langchain_tools",
    "execute_tool_call",
    # LLM
    "CallLLM",
    "LLMRequest",
    "LLMResponse",
    "call_llm_stub",
    "make_openai_llm",
    "OpenAICallLLM",
    "QuizQuestion",
    "LLMValidator",
    "LLMValidatorReport",
    "QuizResult",
    "quiz_report_summary",
    # Human
    "CallHuman",
    "HumanRequest",
    "HumanResponse",
    "ConsoleHuman",
    "MockHuman",
]
