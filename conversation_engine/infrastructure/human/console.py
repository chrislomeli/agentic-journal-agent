"""
ConsoleHuman — CLI-based CallHuman implementation using input().

Simple but functional.  Good for development, debugging, and demos.
"""

from __future__ import annotations

from conversation_engine.infrastructure.human.protocols import (
    HumanRequest,
    HumanResponse,
)


class ConsoleHuman:
    """
    CallHuman implementation that uses the terminal.

    Prints the AI's prompt, then waits for the user to type a response.
    Supports skip via empty input (if allow_skip is True).
    """

    def __init__(self, *, prompt_prefix: str = "You> "):
        self._prefix = prompt_prefix

    def __call__(self, request: HumanRequest) -> HumanResponse:
        print(f"\n{request.prompt}")

        if request.options:
            for i, opt in enumerate(request.options, 1):
                print(f"  [{i}] {opt}")

        raw = input(self._prefix).strip()

        if not raw and request.allow_skip:
            return HumanResponse(content="", skipped=True)

        # If options were provided, resolve numeric choice
        if request.options and raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(request.options):
                return HumanResponse(content=request.options[idx])

        return HumanResponse(content=raw)
