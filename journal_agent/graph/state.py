

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal

STATUS_IDLE = "idle"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"

class JournalState(TypedDict):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    status: Literal["idle", "processing", "completed", "error"]
    error_message: str | None
