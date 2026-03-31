"""SQLite database manager for AIC persistent storage."""
from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consulting_sessions (
    session_id      TEXT PRIMARY KEY,
    problem         TEXT NOT NULL,
    reasoning_summary TEXT,
    validation_status TEXT,
    critique_summary TEXT,
    critique_score  REAL DEFAULT 0.0,
    confidence      REAL DEFAULT 0.0,
    framework_analyses TEXT,
    recommended_frameworks TEXT,
    context         TEXT,
    past_sessions_used INTEGER DEFAULT 0,
    metadata        TEXT,
    created_at      TEXT
);
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    memory_type     TEXT DEFAULT 'episodic',
    importance      REAL DEFAULT 0.5,
    created_at      TEXT DEFAULT (datetime('now')),
    accessed_at     TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    content         TEXT NOT NULL,
    outcome         TEXT,
    confidence      REAL DEFAULT 0.5,
    created_at      TEXT DEFAULT (datetime('now'))
);
"""


class DatabaseManager:
    """Thin wrapper around a SQLite connection."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def create_tables(self) -> None:
        """Create all required tables (idempotent)."""
        with self.get_connection() as conn:
            conn.executescript(_SCHEMA)
        logger.info("Database tables ensured at %s", self._db_path)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield an auto-committing connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
