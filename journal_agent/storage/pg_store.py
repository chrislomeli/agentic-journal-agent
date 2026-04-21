"""pg_store.py — Postgres + pgvector access layer.

Provides PgStore, a psycopg3 connection-pool wrapper with two tiers of helpers:

    Tier 1 — raw SQL primitives:
        fetch_rows()        — SELECT → list[dict]
        execute()           — INSERT / UPDATE / DELETE → rowcount

    Tier 2 — entity upserts (used by the write-through wrappers):
        ensure_session()    — sessions (idempotent)
        upsert_exchanges()  — exchanges (bulk)
        upsert_thread()     — threads + thread_exchanges junction
        upsert_fragment()   — fragments + fragment_exchanges junction (embedding optional)
        upsert_profile()    — user_profiles (single row)

    Plus the pgvector search path:
        search_similar()    — cosine top-k on fragments.embedding

Dependencies (add to pyproject.toml):
    psycopg[binary]>=3.2
    psycopg-pool>=3.2
    pgvector>=0.3
    numpy>=1.26

Schema lives in data/schema.sql — run once:
    psql $POSTGRES_URL -f data/schema.sql
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from journal_agent.configure.settings import get_settings
from journal_agent.model.session import Exchange, Fragment, Tag, ThreadSegment, UserProfile

logger = logging.getLogger(__name__)

# Must match data/schema.sql vector(N). 384 = ChromaDB's default
# sentence-transformer (all-MiniLM-L6-v2). Change both if you swap models.
EMBEDDING_DIM = 384


class PgStore:
    """Postgres + pgvector access layer backed by a connection pool.

    Injected the same way as VectorStore — create once, share across graph nodes.
    """

    def __init__(self, min_size: int = 2, max_size: int = 10):
        url = get_settings().postgres_url
        # open=False defers actual connections until first use so import-time
        # startup doesn't fail if Postgres isn't running.
        self._pool = ConnectionPool(
            conninfo=url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=False,
        )

    def open(self) -> None:
        """Open the pool. Call once at app startup."""
        self._pool.open(wait=True)
        logger.info("PgStore pool open (min=%d max=%d)", self._pool.min_size, self._pool.max_size)

    def close(self) -> None:
        """Drain the pool. Call at app shutdown."""
        self._pool.close()

    @contextmanager
    def _conn(self) -> Generator:
        """Borrow a connection from the pool; psycopg3 auto-commits on clean exit."""
        with self._pool.connection() as conn:
            yield conn

    # ══════════════════════════════════════════════════════════════════════════
    # Tier 1 — raw SQL primitives
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Run a SELECT; return every row as a dict."""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def execute(self, sql: str, params: tuple = ()) -> int:
        """Run a mutating statement; return rowcount."""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    # ══════════════════════════════════════════════════════════════════════════
    # Tier 2 — entity upserts (write-through targets)
    # ══════════════════════════════════════════════════════════════════════════

    def ensure_session(self, session_id: str) -> None:
        """Create a sessions row if it doesn't exist. Safe to call repeatedly."""
        self.execute(
            "INSERT INTO sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (session_id,),
        )

    def upsert_exchanges(self, session_id: str, exchanges: list[Exchange]) -> int:
        """Bulk-upsert a session's Exchange records.

        Ensures the parent sessions row exists first. Returns the rowcount
        from the exchange insert.
        """
        if not exchanges:
            return 0
        self.ensure_session(session_id)

        sql = """
            INSERT INTO exchanges (exchange_id, session_id, timestamp, human_content, ai_content)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (exchange_id) DO UPDATE SET
                timestamp     = EXCLUDED.timestamp,
                human_content = EXCLUDED.human_content,
                ai_content    = EXCLUDED.ai_content
        """
        rows = [
            (
                ex.exchange_id,
                session_id,
                ex.timestamp,
                ex.human.content if ex.human else None,
                ex.ai.content if ex.ai else None,
            )
            for ex in exchanges
        ]
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
            return cur.rowcount

    def upsert_thread(self, session_id: str, thread: ThreadSegment) -> None:
        """Upsert a ThreadSegment + its junction rows.

        Called twice in the normal pipeline: once by save_threads (tags empty)
        and once by save_classified_threads (tags populated). COALESCE on tags
        means a second write with tags overlays the first, but re-running
        save_threads later won't null them out.
        """
        self.ensure_session(session_id)
        thread_id = f"{session_id}:{thread.thread_name}"
        tags_payload = (
            Jsonb([t.model_dump() for t in thread.tags]) if thread.tags else None
        )

        self.execute(
            """
            INSERT INTO threads (thread_id, session_id, thread_name, tags)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE SET
                tags = COALESCE(EXCLUDED.tags, threads.tags)
            """,
            (thread_id, session_id, thread.thread_name, tags_payload),
        )

        # Junction rows. Wipe & replace keeps positions in sync if the thread
        # gets re-decomposed; acceptable because upserts are idempotent per run.
        self.execute(
            "DELETE FROM thread_exchanges WHERE thread_id = %s", (thread_id,)
        )
        if thread.exchange_ids:
            with self._conn() as conn, conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO thread_exchanges (thread_id, exchange_id, position) "
                    "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    [(thread_id, eid, i) for i, eid in enumerate(thread.exchange_ids)],
                )

    def upsert_fragment(
        self,
        fragment: Fragment,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Upsert a Fragment + its junction rows.

        `embedding` is optional — Phase 10 write-through leaves it NULL because
        Chroma is still the source of truth for search. Populate later when
        Postgres takes over retrieval.
        """
        self.ensure_session(fragment.session_id)
        tags_payload = (
            Jsonb([t.model_dump() for t in fragment.tags]) if fragment.tags else None
        )

        self.execute(
            """
            INSERT INTO fragments (fragment_id, session_id, content, tags, embedding, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (fragment_id) DO UPDATE SET
                content   = EXCLUDED.content,
                tags      = EXCLUDED.tags,
                embedding = COALESCE(EXCLUDED.embedding, fragments.embedding),
                timestamp = EXCLUDED.timestamp
            """,
            (
                fragment.fragment_id,
                fragment.session_id,
                fragment.content,
                tags_payload,
                embedding,
                fragment.timestamp,
            ),
        )

        self.execute(
            "DELETE FROM fragment_exchanges WHERE fragment_id = %s",
            (fragment.fragment_id,),
        )
        if fragment.exchange_ids:
            with self._conn() as conn, conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO fragment_exchanges (fragment_id, exchange_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    [(fragment.fragment_id, eid) for eid in fragment.exchange_ids],
                )

    def upsert_profile(self, profile: UserProfile, user_id: str = "default") -> None:
        """Upsert the single user profile row."""
        self.execute(
            """
            INSERT INTO user_profiles (
                user_id, response_style, explanation_depth, tone,
                decision_style, learning_style, interests, pet_peeves,
                active_projects, recurring_themes, human_label, ai_label, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                response_style    = EXCLUDED.response_style,
                explanation_depth = EXCLUDED.explanation_depth,
                tone              = EXCLUDED.tone,
                decision_style    = EXCLUDED.decision_style,
                learning_style    = EXCLUDED.learning_style,
                interests         = EXCLUDED.interests,
                pet_peeves        = EXCLUDED.pet_peeves,
                active_projects   = EXCLUDED.active_projects,
                recurring_themes  = EXCLUDED.recurring_themes,
                human_label       = EXCLUDED.human_label,
                ai_label          = EXCLUDED.ai_label,
                updated_at        = EXCLUDED.updated_at
            """,
            (
                user_id,
                profile.response_style,
                profile.explanation_depth,
                profile.tone,
                profile.decision_style,
                profile.learning_style,
                Jsonb(profile.interests),
                Jsonb(profile.pet_peeves),
                Jsonb(profile.active_projects),
                Jsonb(profile.recurring_themes),
                profile.human_label,
                profile.ai_label,
                profile.updated_at,
            ),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # pgvector search
    # ══════════════════════════════════════════════════════════════════════════

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[tuple[Fragment, float]]:
        """Top-k cosine-similar fragments.

        Uses pgvector's `<=>` cosine-distance operator. Distance ∈ [0, 2];
        we convert to a 0–1 relevance score (`1 - distance / 2`).
        Exchange_ids are reconstructed from the fragment_exchanges junction.
        """
        sql = """
            SELECT
                f.fragment_id,
                f.session_id,
                f.content,
                f.tags,
                f.timestamp,
                COALESCE(
                    (SELECT array_agg(fe.exchange_id)
                     FROM fragment_exchanges fe
                     WHERE fe.fragment_id = f.fragment_id),
                    ARRAY[]::text[]
                ) AS exchange_ids,
                1.0 - (f.embedding <=> %s::vector) / 2.0 AS score
            FROM fragments f
            WHERE f.embedding IS NOT NULL
            ORDER BY f.embedding <=> %s::vector
            LIMIT %s
        """
        rows = self.fetch_rows(sql, (query_embedding, query_embedding, top_k))
        results: list[tuple[Fragment, float]] = []
        for r in rows:
            score = r["score"]
            if score < min_score:
                continue
            raw_tags = r.get("tags")
            tag_data = raw_tags if isinstance(raw_tags, list) else []
            tags = [Tag(**t) if isinstance(t, dict) else t for t in tag_data]
            fragment = Fragment(
                fragment_id=r["fragment_id"],
                session_id=r["session_id"],
                content=r["content"],
                exchange_ids=list(r["exchange_ids"]) if r["exchange_ids"] else [],
                tags=tags,
                timestamp=r["timestamp"] or datetime.now(),
            )
            results.append((fragment, score))
        return results


# ── Module-level singleton (lazy) ──────────────────────────────────────────────

_store: PgStore | None = None


def get_pg_store() -> PgStore:
    """Return the module-level PgStore singleton, opening the pool on first call."""
    global _store
    if _store is None:
        _store = PgStore()
        _store.open()
    return _store
