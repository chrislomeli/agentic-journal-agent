"""checkpointer.py — Async Postgres checkpointer for LangGraph.

The checkpointer persists the full ``JournalState`` between graph super-steps,
keyed by ``thread_id`` (which we set to the session_id).

Lifecycle:
    The async context manager yields a configured AsyncPostgresSaver and tears
    down its connection pool on exit. The checkpointer creates its own async
    connection — separate from the sync PgGateway used by the data layer —
    because LangGraph's async checkpointer requires async psycopg.

Usage:
    async with make_postgres_checkpointer(setup=True) as checkpointer:
        graph = build_journal_graph(..., checkpointer=checkpointer)
        await graph.ainvoke(state, config={"configurable": {"thread_id": sid}})

The ``setup=True`` flag creates the checkpoint tables (idempotent). It is safe
to pass on every run during development; in production, run setup once at
deploy time and pass ``setup=False`` thereafter.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from journal_agent.configure.settings import get_settings


@asynccontextmanager
async def make_postgres_checkpointer(
    setup: bool = False,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Yield an AsyncPostgresSaver bound to the configured Postgres URL.

    Args:
        setup: If True, create checkpoint tables on entry. Idempotent —
            safe to pass repeatedly. Disable in hot paths where the cost of
            a no-op DDL probe matters.
    """
    url = get_settings().postgres_url
    async with AsyncPostgresSaver.from_conn_string(url) as checkpointer:
        if setup:
            await checkpointer.setup()
        yield checkpointer
