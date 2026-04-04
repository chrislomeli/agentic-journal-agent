"""
tool_client — Transport-agnostic tool contracts (MCP-ready).

Architecture:
    LangGraph Node → ToolClient.call(name, args) → LocalToolClient (dev) / McpToolClient (prod)
                                                     ↓
                                                  ToolRegistry → ToolSpec → handler
"""

from conversation_engine.infrastructure.tool_client.spec import ToolSpec
from conversation_engine.infrastructure.tool_client.registry import ToolRegistry
from conversation_engine.infrastructure.tool_client.envelope import (
    ToolContentBlock,
    ToolResultEnvelope,
    ToolResultMeta,
)
from conversation_engine.infrastructure.tool_client.client import (
    ToolClient,
    ToolCallError,
    LocalToolClient,
)
from conversation_engine.infrastructure.tool_client.conversation_tools import (
    AskHumanInput,
    AskHumanOutput,
    RevalidateInput,
    RevalidateOutput,
    MarkCompleteInput,
    MarkCompleteOutput,
    make_ask_human_tool,
    make_revalidate_tool,
    make_mark_complete_tool,
)
from conversation_engine.infrastructure.tool_client.project_graph_tools import (
    ProjectGraphInput,
    ProjectGraphOutput,
    make_project_spec_tool,
    make_project_graph_tool,
)
from conversation_engine.infrastructure.tool_client.langchain_bridge import (
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
    # Conversation tools
    "AskHumanInput",
    "AskHumanOutput",
    "RevalidateInput",
    "RevalidateOutput",
    "MarkCompleteInput",
    "MarkCompleteOutput",
    "make_ask_human_tool",
    "make_revalidate_tool",
    "make_mark_complete_tool",
    # Project spec tools
    "ProjectGraphInput",
    "ProjectGraphOutput",
    "make_project_spec_tool",
    "make_project_graph_tool",
    # LangChain bridge
    "specs_to_langchain_tools",
    "execute_tool_call",
]
