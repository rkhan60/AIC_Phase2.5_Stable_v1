"""ConsultingService — the single facade for all consulting analysis.

This module composes the AutonomousPipeline (goal → reasoning → validation →
self-critique) with the ConsultingFrameworkEngine (SWOT, Porter's, McKinsey 7S,
MECE, Ansoff, BCG) into a single ``analyze()`` call.

All user-facing entry points (main.py, Streamlit, CLI, REST API) should use
this class rather than instantiating the pipeline or framework engine directly.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class ConsultingReport:
    """Structured output of a consulting analysis session."""
    session_id: str
    problem: str
    reasoning_summary: str
    validation_status: str
    critique_summary: str
    critique_score: float
    framework_analyses: Dict[str, Any]
    recommended_frameworks: List[str]
    confidence: float
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    past_sessions_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "problem": self.problem,
            "reasoning_summary": self.reasoning_summary,
            "validation_status": self.validation_status,
            "critique_summary": self.critique_summary,
            "critique_score": self.critique_score,
            "framework_analyses": self.framework_analyses,
            "recommended_frameworks": self.recommended_frameworks,
            "confidence": self.confidence,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "past_sessions_used": self.past_sessions_used,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ConsultingService:
    """Orchestrates the autonomous pipeline and the consulting framework engine.

    Parameters
    ----------
    memory_dir:
        Directory used by the AutonomousPipeline for its memory store.
    db_path:
        Optional path to a SQLite database for persistent session storage.
        When None the service operates in memory-only mode.
    """

    def __init__(self, memory_dir: str = "memory", db_path: Optional[str] = None):
        self._memory_dir = memory_dir
        self._db_path = db_path
        self._pipeline = None          # lazy-initialised (SentenceTransformer is slow)
        self._framework_engine = None
        self._session_repo = None

        # Wire persistent storage if available
        if db_path:
            self._init_storage(db_path)

    # ------------------------------------------------------------------
    # Lazy initialisation helpers
    # ------------------------------------------------------------------

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from .pipeline_autonomy import AutonomousPipeline
            self._pipeline = AutonomousPipeline(memory_dir=self._memory_dir)

    def _ensure_framework_engine(self):
        if self._framework_engine is None:
            from .aic_system import ConsultingFrameworkEngine
            self._framework_engine = ConsultingFrameworkEngine()

    def _init_storage(self, db_path: str):
        try:
            from .storage.session_repository import SessionRepository
            from .storage.database import DatabaseManager
            db = DatabaseManager(db_path)
            db.create_tables()
            self._session_repo = SessionRepository(db)
            logger.info("Persistent session storage enabled at %s", db_path)
        except Exception as exc:
            logger.warning("Could not initialise persistent storage: %s", exc)
            self._session_repo = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def analyze(self, problem: str, context: Optional[Dict[str, Any]] = None) -> ConsultingReport:
        """Run a full consulting analysis and return a structured report.

        Steps:
          1. Retrieve relevant past sessions (if storage is enabled).
          2. Run the autonomous pipeline (goal planning → reasoning → validation
             → self-critique).
          3. Run the consulting framework engine.
          4. Combine results into a ConsultingReport.
          5. Persist the session (if storage is enabled).
        """
        if not problem or not problem.strip():
            raise ValueError("problem must be a non-empty string")

        context = dict(context or {})
        session_id = uuid.uuid4().hex[:16]
        past_sessions_used = 0

        # ---- 1. Past sessions context --------------------------------
        if self._session_repo:
            try:
                past = self._session_repo.find_similar(problem, limit=3)
                if past:
                    context["past_sessions"] = [
                        {"problem": s.get("problem", ""), "summary": s.get("reasoning_summary", "")}
                        for s in past
                    ]
                    past_sessions_used = len(past)
            except Exception as exc:
                logger.warning("Could not load past sessions: %s", exc)

        # ---- 2. Autonomous pipeline ----------------------------------
        reasoning_summary = "Pipeline not executed"
        validation_status = "unknown"
        critique_summary = "No critique"
        critique_score = 0.0
        pipeline_confidence = 0.5

        try:
            self._ensure_pipeline()
            result = self._pipeline.run_cycle(problem, context)
            reasoning_summary = result.reasoning_summary
            validation_status = result.validation_status
            critique_summary = result.self_critic_summary
            # AutonomousResult stores overall_score in metadata
            critique_score = float(result.metadata.get("critique_score", 0.0))
            pipeline_confidence = 0.8 if validation_status == "valid" else 0.4
        except Exception as exc:
            logger.error("Autonomous pipeline error: %s", exc, exc_info=True)

        # ---- 3. Framework analysis -----------------------------------
        framework_analyses: Dict[str, Any] = {}
        recommended_frameworks: List[str] = []

        try:
            self._ensure_framework_engine()
            business_ctx = self._build_business_context(context)
            recommended_frameworks = self._framework_engine.recommend_frameworks(business_ctx)
            framework_analyses = self._framework_engine.analyse(problem, business_ctx)
        except Exception as exc:
            logger.error("Framework engine error: %s", exc, exc_info=True)

        # ---- 4. Combine --------------------------------------------
        confidence = round((pipeline_confidence + (critique_score or 0.5)) / 2, 4)

        report = ConsultingReport(
            session_id=session_id,
            problem=problem,
            reasoning_summary=reasoning_summary,
            validation_status=validation_status,
            critique_summary=critique_summary,
            critique_score=critique_score,
            framework_analyses=framework_analyses,
            recommended_frameworks=recommended_frameworks,
            confidence=confidence,
            context={k: v for k, v in context.items() if k != "past_sessions"},
            past_sessions_used=past_sessions_used,
        )

        # ---- 5. Persist --------------------------------------------
        if self._session_repo:
            try:
                self._session_repo.save_session(report.to_dict())
            except Exception as exc:
                logger.warning("Could not persist session: %s", exc)

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_business_context(self, context: Dict[str, Any]):
        """Build a BusinessContext dataclass from a free-form context dict."""
        from .aic_system import BusinessContext
        return BusinessContext(
            industry=context.get("industry", "general"),
            company_size=context.get("company_size", "sme"),
            market_position=context.get("market_position", "challenger"),
            competitive_landscape=context.get("competitive_landscape", {}),
            financial_health=context.get("financial_health", {}),
            organizational_maturity=context.get("organizational_maturity", "developing"),
            strategic_priorities=context.get("strategic_priorities", []),
            stakeholder_map=context.get("stakeholder_map", {}),
            cultural_context=context.get("cultural_context", ""),
            regulatory_environment=context.get("regulatory_environment", {}),
        )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a past session by ID (requires persistent storage)."""
        if self._session_repo:
            return self._session_repo.get_session(session_id)
        return None

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent sessions (requires persistent storage)."""
        if self._session_repo:
            return self._session_repo.list_sessions(limit=limit)
        return []
