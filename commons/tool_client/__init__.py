"""
tool_client — Transport-agnostic tool contracts (MCP-ready).

Architecture:
    LangGraph Node → ToolClient.call(name, args) → LocalToolClient (dev) / McpToolClient (prod)
                                                     ↓
                                                  ToolRegistry → ToolSpec → handler

This package contains only the generic tool infrastructure.
Domain-specific tool definitions (ask_human, revalidate, etc.) belong
in the consuming application, not here.
"""

from commons.tool_client.spec import ToolSpec
from commons.tool_client.registry import ToolRegistry
from commons.tool_client.envelope import (
    ToolContentBlock,
    ToolResultEnvelope,
    ToolResultMeta,
)
from commons.tool_client.client import (
    ToolClient,
    ToolCallError,
    LocalToolClient,
)
from commons.tool_client.langchain_bridge import (
    specs_to_langchain_tools,
    execute_tool_call,
)

__all__ = [
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
]
