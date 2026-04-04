"""
ToolClient — Abstract interface + LocalToolClient implementation.

LangGraph nodes call ToolClient.call(name, args).
Every call returns a ToolResultEnvelope with provenance metadata.
During dev, LocalToolClient dispatches in-process.
Later, swap in McpToolClient without changing graph logic.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from conversation_engine.infrastructure.tool_client.envelope import (
    ToolContentBlock,
    ToolResultEnvelope,
    ToolResultMeta,
)
from conversation_engine.infrastructure.tool_client.registry import ToolRegistry


class ToolCallError(Exception):
    """Raised when a tool call fails validation or execution."""

    def __init__(self, tool_name: str, kind: str, details: Any):
        self.tool_name = tool_name
        self.kind = kind
        self.details = details
        super().__init__(f"[{tool_name}] {kind}: {details}")


class ToolClient(ABC):
    """Abstract tool client — the only interface LangGraph nodes should use."""

    @abstractmethod
    def call(self, tool_name: str, args: dict[str, Any]) -> ToolResultEnvelope:
        """Call a tool by name with JSON-safe args, return envelope with metadata + payload."""
        ...

    @abstractmethod
    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP-style tool definitions (name/description/inputSchema/outputSchema)."""
        ...


class LocalToolClient(ToolClient):
    """
    In-process tool client for local development.

    Validates inputs and outputs via Pydantic models,
    enforcing the same boundary discipline as a remote MCP call.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def list_tools(self) -> list[dict[str, Any]]:
        return self._registry.catalog()

    def call(self, tool_name: str, args: dict[str, Any]) -> ToolResultEnvelope:
        # Lookup — raises KeyError (programmer bug, not LLM-recoverable)
        tool = self._registry.get(tool_name)
        input_hash = ToolResultMeta.hash_args(args)
        t0 = time.perf_counter()

        # Validate input — in-band error (LLM can retry with different args)
        try:
            validated_input = tool.input_model(**args)
        except ValidationError as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            details = e.errors()
            msg = f"[{tool_name}] input_validation_error: {details}"
            return ToolResultEnvelope(
                meta=ToolResultMeta(
                    tool_name=tool_name,
                    tool_description=tool.description,
                    input_args=args,
                    input_hash=input_hash,
                    duration_ms=round(duration_ms, 3),
                    success=False,
                    error="input_validation_error",
                ),
                is_error=True,
                content=[ToolContentBlock(text=msg)],
                structured={"kind": "input_validation_error", "details": details},
            )

        # Re-hash validated input for better determinism
        try:
            canonical = validated_input.model_dump_json(sort_keys=True)
            input_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        except Exception:
            pass  # keep earlier hash

        # Execute handler — in-band error (LLM can see failure and adapt)
        try:
            result = tool.handler(validated_input)
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            msg = f"[{tool_name}] execution_error: {e}"
            return ToolResultEnvelope(
                meta=ToolResultMeta(
                    tool_name=tool_name,
                    tool_description=tool.description,
                    input_args=args,
                    input_hash=input_hash,
                    duration_ms=round(duration_ms, 3),
                    success=False,
                    error="execution_error",
                ),
                is_error=True,
                content=[ToolContentBlock(text=msg)],
                structured={"kind": "execution_error", "details": str(e)},
            )

        # Validate output (accepts model or dict)
        try:
            validated_output = tool.output_model.model_validate(result)
        except ValidationError as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            details = e.errors()
            msg = f"[{tool_name}] output_validation_error: {details}"
            return ToolResultEnvelope(
                meta=ToolResultMeta(
                    tool_name=tool_name,
                    tool_description=tool.description,
                    input_args=args,
                    input_hash=input_hash,
                    duration_ms=round(duration_ms, 3),
                    success=False,
                    error="output_validation_error",
                ),
                is_error=True,
                content=[ToolContentBlock(text=msg)],
                structured={"kind": "output_validation_error", "details": details},
            )

        duration_ms = (time.perf_counter() - t0) * 1000
        structured = validated_output.model_dump()
        content_text = json.dumps(structured, indent=2, sort_keys=True)

        return ToolResultEnvelope(
            meta=ToolResultMeta(
                tool_name=tool_name,
                tool_description=tool.description,
                input_args=args,
                input_hash=input_hash,
                duration_ms=round(duration_ms, 3),
                success=True,
            ),
            structured=structured,
            content=[ToolContentBlock(text=content_text)],
        )
