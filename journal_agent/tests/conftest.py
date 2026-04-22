"""Shared test fixtures and stubs.

The psycopg stub must run before any test module tries to import
repositories → pg_gateway, which pulls in psycopg at module level.
"""

import importlib.util
import sys
from unittest.mock import MagicMock

# ── psycopg stub (needed when psycopg is not installed) ──────────────────────
if importlib.util.find_spec("psycopg") is None:
    _stub = MagicMock()
    _stub.__path__ = []  # make it a package so submodules import
    _stub.rows = MagicMock()
    _stub.rows.dict_row = MagicMock()
    _stub.types = MagicMock()
    _stub.types.json = MagicMock()
    _stub.types.json.Jsonb = MagicMock()
    sys.modules.setdefault("psycopg", _stub)
    sys.modules.setdefault("psycopg.rows", _stub.rows)
    sys.modules.setdefault("psycopg.types", _stub.types)
    sys.modules.setdefault("psycopg.types.json", _stub.types.json)

if importlib.util.find_spec("psycopg_pool") is None:
    _pool_stub = MagicMock()
    _pool_stub.ConnectionPool = MagicMock()
    sys.modules.setdefault("psycopg_pool", _pool_stub)

# ── fastembed stub (needed when fastembed is not installed) ───────────────────
if importlib.util.find_spec("fastembed") is None:
    _fe_stub = MagicMock()
    _fe_stub.TextEmbedding = MagicMock()
    sys.modules.setdefault("fastembed", _fe_stub)
