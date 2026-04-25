"""SSE streaming helpers.

An SSE event over HTTP looks like this on the wire:

    event: token
    data: Hello

    event: token
    data:  world

    event: done
    data:

Each field ends with \\n, and events are separated by a blank line (\\n\\n).
The browser's EventSource API (or any SSE client) parses this automatically.
"""
import asyncio
import json
from collections.abc import AsyncGenerator

from journal_agent.api.models import SseEvent


def _sse(event: SseEvent, data: str) -> str:
    """Format one SSE event as a wire string."""
    payload = json.dumps({"text": data})
    return f"event: {event}\ndata: {payload}\n\n"


async def fake_stream(message: str) -> AsyncGenerator[str, None]:
    """Fake slow AI response — yields one token at a time with a short delay.

    This exists purely to verify that SSE streaming works end-to-end before
    wiring in LangGraph. Replace with graph_stream() when ready.
    """
    words = f"You said: '{message}'. This is a fake streaming response.".split()
    for word in words:
        yield _sse(SseEvent.TOKEN, word + " ")
        await asyncio.sleep(0.7)
    yield _sse(SseEvent.DONE, "")
