"""
ToolResultEnvelope — Every tool call returns its payload wrapped in metadata.

This is enforced at the ToolClient layer, not by the tool itself.
The tool handler stays pure (input → output). The client wraps the result.

The envelope tells downstream consumers (e.g. StateMediator):
  - Which tool produced this result
  - What inputs were provided
  - When it ran and how long it took
  - Whether it succeeded or failed

JSON-serializable by design — compatible with MCP transport.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolResultMeta(BaseModel):
    """Metadata about a tool invocation. Attached to every result."""

    tool_name: str
    tool_description: str = ""
    input_args: dict[str, Any]
    input_hash: str = Field(
        default="",
        description="SHA-256 hex digest of canonical JSON input — for dedup/caching",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the call completed",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Wall-clock execution time in milliseconds",
    )
    success: bool = True
    error: str | None = None

    @staticmethod
    def hash_args(args: dict[str, Any]) -> str:
        """Deterministic hash of input args for caching/dedup."""
        canonical = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ToolContentBlock(BaseModel):
    """Minimal MCP-compatible content block. Expand later for images/audio/etc."""

    type: Literal["text"] = "text"
    text: str


class ToolResultEnvelope(BaseModel):
    """
    Wraps every tool result with provenance metadata.

    MCP-aligned shape:
      - content: MCP-style ContentBlock[] (always present)
      - structured: optional structured payload (maps to MCP structuredContent)
      - is_error: tool-level failure indicator (maps to MCP isError)

    Nodes and middleware can inspect .meta to route, log, or
    augment state based on which tool produced the result.
    """

    meta: ToolResultMeta
    content: list[ToolContentBlock] = Field(default_factory=list)
    structured: dict[str, Any] | None = None
    is_error: bool = False

    @property
    def payload(self) -> dict[str, Any]:
        """Backward-compat alias — returns structured or empty dict."""
        return self.structured or {}

    def model_dump_flat(self) -> dict[str, Any]:
        """Return structured payload with _meta key injected — for embedding in state dicts."""
        base = self.structured or {}
        return {**base, "_meta": self.meta.model_dump()}
