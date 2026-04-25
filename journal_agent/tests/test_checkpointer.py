"""Tests for checkpointer wiring (design/api-build-plan.md item #4).

We exercise the persistence contract with ``MemorySaver`` — the in-memory
checkpointer — to verify behavior without touching Postgres. The mechanic
is identical to ``AsyncPostgresSaver``; only the storage backend differs.

What the tests prove:
    1. With the same thread_id, state accumulates across separate invocations.
       This is the per-turn HTTP request pattern: each request runs the graph
       to END and the checkpointer holds in-flight state between requests.
    2. Different thread_ids have isolated state — sessions don't bleed into
       each other.
    3. The checkpointer's stored state is retrievable via ``aget_state`` —
       useful for inspection, debugging, and the future "load history" API.
"""

from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


# ── Test fixtures ────────────────────────────────────────────────────────────


class _MiniState(BaseModel):
    """Minimal state schema mirroring JournalState.session_messages.

    Uses the same ``add_messages`` reducer so the test exercises the same
    merge semantics our real graph relies on.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: Annotated[list, add_messages] = Field(default_factory=list)


def _build_mini_graph(checkpointer):
    """Build a one-node graph that echoes each HumanMessage as an AIMessage."""

    def respond(state: _MiniState) -> dict:
        last_human = next(
            (m for m in reversed(state.messages) if isinstance(m, HumanMessage)),
            None,
        )
        text = f"echo: {last_human.content}" if last_human else "echo: (nothing)"
        return {"messages": [AIMessage(content=text)]}

    builder = StateGraph(_MiniState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)


# ── Tests ────────────────────────────────────────────────────────────────────


async def test_checkpoint_persists_state_across_invocations():
    """Same thread_id, two ainvokes — messages should accumulate.

    Each ainvoke is a fresh "turn." The graph picks up prior state from the
    checkpointer, applies the new HumanMessage via add_messages, and runs
    one node. After two turns the message list should hold both rounds.
    """
    checkpointer = MemorySaver()
    graph = _build_mini_graph(checkpointer)
    config = {"configurable": {"thread_id": "session-A"}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hello")]},
        config=config,
    )

    final_state = await graph.ainvoke(
        {"messages": [HumanMessage(content="again")]},
        config=config,
    )

    contents = [m.content for m in final_state["messages"]]
    assert contents == ["hello", "echo: hello", "again", "echo: again"]


async def test_checkpoint_isolates_thread_ids():
    """Different thread_ids must not see each other's state."""
    checkpointer = MemorySaver()
    graph = _build_mini_graph(checkpointer)

    await graph.ainvoke(
        {"messages": [HumanMessage(content="from-A")]},
        config={"configurable": {"thread_id": "session-A"}},
    )

    state_b = await graph.ainvoke(
        {"messages": [HumanMessage(content="from-B")]},
        config={"configurable": {"thread_id": "session-B"}},
    )

    contents_b = [m.content for m in state_b["messages"]]
    assert contents_b == ["from-B", "echo: from-B"]


async def test_checkpoint_state_is_retrievable():
    """``aget_state`` should return the saved snapshot for a thread_id.

    This is the path a future "show me my conversation history" API would
    use — read the checkpointer to see in-flight state without re-running
    the graph.
    """
    checkpointer = MemorySaver()
    graph = _build_mini_graph(checkpointer)
    config = {"configurable": {"thread_id": "session-A"}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hello")]},
        config=config,
    )

    snapshot = await graph.aget_state(config)
    assert snapshot is not None
    contents = [m.content for m in snapshot.values["messages"]]
    assert contents == ["hello", "echo: hello"]


async def test_unknown_thread_id_starts_clean():
    """A thread_id with no checkpoint should start with empty state."""
    checkpointer = MemorySaver()
    graph = _build_mini_graph(checkpointer)

    state = await graph.ainvoke(
        {"messages": [HumanMessage(content="first words")]},
        config={"configurable": {"thread_id": "brand-new"}},
    )

    contents = [m.content for m in state["messages"]]
    assert contents == ["first words", "echo: first words"]
