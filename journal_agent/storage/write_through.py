"""write_through.py — Store-level write-through wrappers.

Each wrapper is a drop-in for an existing local store: same protocol, same
method signatures. Callers see one interface; the wrapper fans writes out to
JSONL (or Chroma) for durability AND to Postgres for queryability.

Pairings:
    WriteThroughTranscriptStore  satisfies ArtifactStore  — Exchange records
    WriteThroughThreadStore      satisfies ArtifactStore  — ThreadSegment records
                                                            (same class is reused for
                                                             the threads/ and
                                                             classified_threads/ folders;
                                                             the PG table is merged)
    WriteThroughFragmentStore    satisfies FragmentStore  — Fragment records
    WriteThroughProfileStore     satisfies ProfileStore   — UserProfile

Read paths (load_*, search_*, get_last_session_id) delegate to the local store —
JSONL/Chroma stay authoritative during the transition. Flip individual reads to
Postgres once you trust parity.

Enable in main.py via settings.enable_postgres. When False, use the plain stores
as before; when True, wrap them here.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from journal_agent.model.session import Exchange, Fragment, ThreadSegment, UserProfile
from journal_agent.storage.chroma_fragment_store import ChromaFragmentStore
from journal_agent.storage.pg_store import PgStore
from journal_agent.storage.profile_store import UserProfileStore
from journal_agent.storage.storage import JsonStore

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class WriteThroughTranscriptStore:
    """ArtifactStore for Exchange records: JSONL + Postgres exchanges table."""

    def __init__(self, json_store: JsonStore, pg_store: PgStore):
        self._json = json_store
        self._pg = pg_store

    def save_session(self, session_id: str, items: list[Exchange]) -> None:
        self._json.save_session(session_id, items)
        self._pg.upsert_exchanges(session_id, items)

    def load_session(self, session_id: str, model: type[T] = Exchange) -> list[T] | None:
        return self._json.load_session(session_id, model)

    def get_last_session_id(self) -> str | None:
        return self._json.get_last_session_id()


class WriteThroughThreadStore:
    """ArtifactStore for ThreadSegment records: JSONL + Postgres threads table.

    Used for both the `threads/` folder (pre-classification, tags empty) and
    the `classified_threads/` folder (tags populated). The PG table is merged;
    the upsert COALESCEs tags so the classified write overlays the unclassified
    one without losing tags on re-runs.
    """

    def __init__(self, json_store: JsonStore, pg_store: PgStore):
        self._json = json_store
        self._pg = pg_store

    def save_session(self, session_id: str, items: list[ThreadSegment]) -> None:
        self._json.save_session(session_id, items)
        for thread in items:
            self._pg.upsert_thread(session_id, thread)

    def load_session(self, session_id: str, model: type[T] = ThreadSegment) -> list[T] | None:
        return self._json.load_session(session_id, model)

    def get_last_session_id(self) -> str | None:
        return self._json.get_last_session_id()


class WriteThroughFragmentStore:
    """FragmentStore: Chroma (embeddings + search) + Postgres (content + junctions).

    Embeddings stay in Chroma for now; Postgres stores content/metadata with
    embedding=NULL. Search still goes through Chroma. Once you're ready to
    move search to pgvector, generate embeddings here and pass them to
    pg.upsert_fragment(fragment, embedding=vec).
    """

    def __init__(self, chroma_store: ChromaFragmentStore, pg_store: PgStore):
        self._chroma = chroma_store
        self._pg = pg_store

    def save_fragments(self, fragments: list[Fragment]) -> None:
        if not fragments:
            return
        self._chroma.save_fragments(fragments)
        for f in fragments:
            self._pg.upsert_fragment(f, embedding=None)

    def search_fragments(
        self,
        query_text: str,
        min_relevance: float = 0.3,
        top_k: int = 5,
    ) -> list[tuple[Fragment, float]]:
        return self._chroma.search_fragments(query_text, min_relevance, top_k)

    def load_all(self, user_id: str | None = None) -> list[Fragment]:
        return self._chroma.load_all(user_id)


class WriteThroughProfileStore:
    """ProfileStore: local JSON + Postgres user_profiles row."""

    def __init__(self, local_store: UserProfileStore, pg_store: PgStore):
        self._local = local_store
        self._pg = pg_store

    def load_profile(self, user_id: str | None = None) -> UserProfile | None:
        return self._local.load_profile(user_id)

    def save_profile(self, profile: UserProfile, user_id: str | None = None) -> None:
        self._local.save_profile(profile, user_id)
        self._pg.upsert_profile(profile, user_id or "default")
