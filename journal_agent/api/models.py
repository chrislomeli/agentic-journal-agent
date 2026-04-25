"""API request and response models.

These are the shapes that cross the HTTP boundary — separate from the internal
JournalState so the API contract can evolve independently of the graph.
"""
from enum import StrEnum

from pydantic import BaseModel


class MessageRole(StrEnum):
    USER = "user"
    AI = "ai"
    SYSTEM = "system"


class ChatRequest(BaseModel):
    message: str


class SseEvent(StrEnum):
    """SSE event type names sent in the `event:` field."""
    TOKEN = "token"       # one chunk of AI text
    SYSTEM = "system"     # feedback message (e.g. /save confirmation)
    DONE = "done"         # stream complete
    ERROR = "error"       # something went wrong



