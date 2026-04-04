"""
ToolRegistry — Central catalog of available tools.

Supports registration, lookup, listing, and JSON Schema export
for future MCP server exposure.
"""

from __future__ import annotations

from typing import Any

from conversation_engine.infrastructure.tool_client.spec import ToolSpec


class ToolRegistry:
    """Thread-safe tool registry."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        """Register a tool. Raises if name already taken."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        """Look up a tool by name. Raises KeyError if not found."""
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise KeyError(f"Tool '{name}' not found. Available: {available}") from None

    def list_tools(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def catalog(self) -> list[dict[str, Any]]:
        """Return MCP-style tool catalog (name, description, inputSchema, outputSchema)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema(),
                "outputSchema": t.output_schema(),
            }
            for t in self._tools.values()
        ]
