"""
human — Backwards-compatibility shim.

Reusable human protocols have moved to ``commons.human``.
"""

from commons.human import (
    CallHuman,
    HumanRequest,
    HumanResponse,
    ConsoleHuman,
    MockHuman,
)

__all__ = [
    "CallHuman",
    "HumanRequest",
    "HumanResponse",
    "ConsoleHuman",
    "MockHuman",
]
