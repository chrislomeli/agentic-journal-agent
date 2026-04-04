"""
human — Protocol-based human interaction layer.

Protocols
---------
CallHuman     Surface a message to a human, get their response back.

Implementations
---------------
ConsoleHuman    CLI-based (input/print).
MockHuman       Deterministic stub for testing.

Swap in any callable that matches the Protocol signature.
"""

from conversation_engine.infrastructure.human.protocols import (
    CallHuman,
    HumanRequest,
    HumanResponse,
)
from conversation_engine.infrastructure.human.console import ConsoleHuman
from conversation_engine.infrastructure.human.mock import MockHuman

__all__ = [
    "CallHuman",
    "HumanRequest",
    "HumanResponse",
    "ConsoleHuman",
    "MockHuman",
]
