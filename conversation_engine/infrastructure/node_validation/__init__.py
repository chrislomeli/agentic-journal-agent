"""
node_validation — Structured result envelope + input validation for LangGraph nodes.

- NodeResult — structured ok/error envelope every node can return
- validated_node — decorator that validates state against a Pydantic schema
- handle_error — generic error-handler node that reads NodeResult and marks failure
"""

from conversation_engine.infrastructure.node_validation.result_schema import (
    NodeError,
    NodeResult,
)
from conversation_engine.infrastructure.node_validation.validator_decorator import (
    validated_node,
)
from conversation_engine.infrastructure.node_validation.handle_error import (
    handle_error,
)

__all__ = [
    "NodeError",
    "NodeResult",
    "validated_node",
    "handle_error",
]
