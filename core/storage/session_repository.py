"""Session repository — CRUD and similarity search for consulting sessions."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class SessionRepository:
    """Persist and retrieve ConsultingReport snapshots in SQLite."""

    def __init__(self, db: DatabaseManager):
        self._db = db

    def save_session(self, report: Dict[str, Any]) -> None:
        sql = """
            INSERT OR REPLACE INTO consulting_sessions (
                session_id, problem, reasoning_summary, validation_status,
                critique_summary, critique_score, confidence,
                framework_analyses, recommended_frameworks, context,
                past_sessions_used, metadata, created_at
            ) VALUES (
                :session_id, :problem, :reasoning_summary, :validation_status,
                :critique_summary, :critique_score, :confidence,
                :framework_analyses, :recommended_frameworks, :context,
                :past_sessions_used, :metadata, :created_at
            )
        """
        params = {
            "session_id": report["session_id"],
            "problem": report["problem"],
            "reasoning_summary": report.get("reasoning_summary", ""),
            "validation_status": report.get("validation_status", "unknown"),
            "critique_summary": report.get("critique_summary", ""),
            "critique_score": report.get("critique_score", 0.0),
            "confidence": report.get("confidence", 0.0),
            "framework_analyses": json.dumps(report.get("framework_analyses", {})),
            "recommended_frameworks": json.dumps(
                [str(f) for f in report.get("recommended_frameworks", [])]
            ),
            "context": json.dumps(report.get("context", {})),
            "past_sessions_used": report.get("past_sessions_used", 0),
            "metadata": json.dumps(report.get("metadata", {})),
            "created_at": report.get("created_at", ""),
        }
        with self._db.get_connection() as conn:
            conn.execute(sql, params)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM consulting_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._deserialise(row) if row else None

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM consulting_sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._deserialise(r) for r in rows]

    def find_similar(self, problem: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Keyword-overlap similarity search (no embeddings required)."""
        query_tokens = {
            w for w in problem.lower().split() if len(w) >= 4 and w.isalpha()
        }
        if not query_tokens:
            return []
        with self._db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM consulting_sessions ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
        scored = []
        for row in rows:
            doc_tokens = {
                w for w in (row["problem"] or "").lower().split()
                if len(w) >= 4 and w.isalpha()
            }
            overlap = len(query_tokens & doc_tokens)
            if overlap > 0:
                scored.append((overlap, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._deserialise(r) for _, r in scored[:limit]]

    @staticmethod
    def _deserialise(row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("framework_analyses", "recommended_frameworks", "context", "metadata"):
            if isinstance(d.get(key), str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key] = {}
        return d
