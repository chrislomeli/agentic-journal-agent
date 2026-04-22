"""write_through.py — Store-level write-through wrappers for non-fragment stores.
In the future we may want to write JSON records to a queue, and have a consumer load them into Postgres. (store-forward patter)
But for now we just perform both writes in the same process.

Fragment storage is handled by PgFragmentStore (pg_fragment_store.py) when
Postgres is enabled — no write-through needed there.

Remaining wrappers:
    WriteThroughTranscriptStore  satisfies ArtifactStore  — Exchange records
    WriteThroughThreadStore      satisfies ArtifactStore  — ThreadSegment records
                                                            (reused for threads/ and
                                                             classified_threads/ folders;
                                                             PG table is merged)
    WriteThroughProfileStore     satisfies ProfileStore   — UserProfile

Read paths delegate to the local store (JSONL stays authoritative during transition).
Enable via settings.enable_postgres in main.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from journal_agent.model.session import Exchange, Role, ThreadSegment, Turn, UserProfile, Insight
from journal_agent.storage.jsonl_gateway import JsonlGateway
from journal_agent.storage.pg_gateway import PgGateway

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ── Shared conversion (used by both JSONL and Postgres read paths) ───────────


def exchanges_to_messages(exchanges: list[Exchange]) -> list[BaseMessage]:
    """Convert Exchange records to a flat list of LangChain messages.

    Ordering: for each exchange, human turn first, then AI turn.
    """
    messages: list[BaseMessage] = []
    for exchange in exchanges:
        if exchange.human:
            if exchange.human.role == Role.HUMAN:
                messages.append(HumanMessage(content=exchange.human.content))
            elif exchange.human.role == Role.SYSTEM:
                messages.append(SystemMessage(content=exchange.human.content))
        if exchange.ai:
            messages.append(AIMessage(content=exchange.ai.content))
    return messages


class TranscriptRepository:
    """ArtifactStore for Exchange records: writes to JSONL + Postgres; reads from Postgres."""

    def __init__(self, jsonl_gateway: JsonlGateway, pg_gateway: PgGateway):
        self._jsonl = jsonl_gateway
        self._pg = pg_gateway

    def save_collection(self, name: str, items: list[Exchange]) -> None:
        self._jsonl.save_json(name, items)
        self._pg.upsert_exchanges(name, items)

    def load_collection(self, name: str, model: type[T] = Exchange) -> list[T] | None:
        rows = self._pg.fetch_exchanges(name)
        return rows or None  # protocol: return None on miss, not []

    def get_last_session_id(self) -> str | None:
        return self._pg.get_last_session_id()

    def retrieve_transcript(self) -> list[BaseMessage] | None:
        """Load the most recent saved session as LangChain messages, or None."""
        latest = self.get_last_session_id()
        if latest is None:
            return None
        exchanges = self.load_collection(latest)
        if exchanges is None:
            return None
        return exchanges_to_messages(exchanges)


class ThreadsRepository:
    """ArtifactStore for ThreadSegment records: writes to JSONL + Postgres; reads from Postgres.

    Reused for both threads/ and classified_threads/ folders — the PG table is merged;
    upsert COALESCEs tags so classification overlays the initial write.
    """

    def __init__(self, jsonl_gateway: JsonlGateway, pg_gateway: PgGateway):
        self._jsonl = jsonl_gateway
        self._pg = pg_gateway

    def save_collection(self, name: str, items: list[ThreadSegment]) -> None:
        self._jsonl.save_json(name, items)
        for thread in items:
            self._pg.upsert_thread(name, thread)

    def load_collection(self, name: str, model: type[T] = ThreadSegment) -> list[T] | None:
        rows = self._pg.fetch_threads(name)
        return rows or None  # protocol: return None on miss, not []

    def get_last_session_id(self) -> str | None:
        return self._pg.get_last_session_id()


class UserProfileRepository:
    """ProfileStore: local JSON + Postgres user_profiles row."""

    def __init__(self,  jsonl_gateway: JsonlGateway, pg_gateway: PgGateway):
        self._json = jsonl_gateway
        self._pg = pg_gateway

    def load_profile(self, user_id: str | None = "default_user") -> UserProfile | None:
        user_profile = self._pg.fetch_profile(user_id)
        user_profile.is_current = True
        user_profile.is_updated = False
        return user_profile

    def save_profile(self, profile: UserProfile) -> None:
        self._json.save_json(profile.user_id, [profile])
        self._pg.upsert_profile(profile)


class InsightsRepository:
    """ArtifactStore for Intent records: writes to JSONL + Postgres; reads from Postgres."""

    def __init__(self, jsonl_gateway: JsonlGateway, pg_gateway: PgGateway):
        self._jsonl = jsonl_gateway
        self._pg = pg_gateway

    def save_insights(self, items: list[Insight]) -> None:
        if len(items) > 0:
            from datetime import datetime
            file_name = f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._jsonl.save_json(file_name, items)
            self._pg.upsert_insights(items)

    def load_insights(self, search_label: str | None = None, date_cutoff: datetime | None = None ) -> list[T] | None:
        rows = self._pg.fetch_insights(search_label, date_cutoff)
        return rows or None  # protocol: return None on miss, not []
