from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from journal_agent.model.session import Fragment, Exchange, ThreadSegment, ContextSpecification, Status


class JournalState(TypedDict):
    session_id: str
    recent_messages: list[BaseMessage]
    session_messages: Annotated[list[BaseMessage], add_messages]
    transcript: Annotated[list[Exchange], add]
    threads: Annotated[list[ThreadSegment], add]
    classified_threads: Annotated[list[ThreadSegment], add]
    fragments: list[Fragment]  # new — written by classify, read by extract
    retrieved_history: list[Fragment]  # lookup by user query
    context_specification: ContextSpecification  # current instruction from intent classification
    status: Status
    error_message: str | None
