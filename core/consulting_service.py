"""ConsultingService — main facade for the AIC product."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConsultingReport:
    """Structured output from a consulting analysis run."""
    session_id: str
    problem: str
    reasoning_summary: str
    validation_status: str
    critique_summary: str
    critique_score: float
    framework_analyses: Dict[str, Any]
    recommended_frameworks: List[Any]
    confidence: float
    context: Dict[str, Any]
    created_at: datetime
    past_sessions_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


class ConsultingService:
    """Facade composing AutonomousPipeline + ConsultingFrameworkEngine.

    Lazy initialisation: heavy components are created on first call to
    ``analyze()``, not at import time.
    """

    def __init__(
        self,
        memory_dir: str = "memory",
        db_path: Optional[str] = None,
    ):
        self._memory_dir = memory_dir
        self._db_path = db_path
        self._pipeline = None
        self._framework_engine = None
        self._session_repo = None

    # ------------------------------------------------------------------
    # Initialisation (lazy)
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        if self._framework_engine is not None:
            return

        from .aic_system import ConsultingFrameworkEngine
        self._framework_engine = ConsultingFrameworkEngine()

        # Autonomous pipeline requires SentenceTransformer model download;
        # degrade gracefully in air-gapped / offline environments.
        try:
            from .pipeline_autonomy import AutonomousPipeline
            self._pipeline = AutonomousPipeline(memory_dir=self._memory_dir)
            logger.info("AutonomousPipeline initialised.")
        except Exception as exc:
            logger.warning(
                "AutonomousPipeline unavailable (%s) — using framework-only mode.", exc
            )
            self._pipeline = None

        if self._db_path:
            try:
                from .storage.database import DatabaseManager
                from .storage.session_repository import SessionRepository
                db = DatabaseManager(self._db_path)
                db.create_tables()
                self._session_repo = SessionRepository(db)
                logger.info("Persistent storage enabled at %s", self._db_path)
            except Exception as exc:
                logger.warning("Storage init failed (%s) — running without persistence.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConsultingReport:
        """Run a full consulting analysis.

        1. Retrieve similar past sessions for context enrichment.
        2. Run the autonomous reasoning pipeline.
        3. Apply framework analyses.
        4. Persist the result.
        """
        if not problem or not problem.strip():
            raise ValueError("problem must not be empty")

        self._ensure_init()
        context = context or {}
        session_id = str(uuid.uuid4())

        # 1. Past sessions
        past_sessions: List[Dict] = []
        if self._session_repo:
            try:
                past_sessions = self._session_repo.find_similar(problem, limit=3)
            except Exception as exc:
                logger.warning("Could not retrieve past sessions: %s", exc)

        # Enrich pipeline context with past learning
        pipeline_context: Dict[str, Any] = dict(context)
        if past_sessions:
            pipeline_context["past_sessions"] = [
                {"problem": s.get("problem"), "summary": s.get("reasoning_summary")}
                for s in past_sessions[:3]
            ]

        # 2. Autonomous pipeline (may be None in offline/degraded mode)
        cycle_result = None
        if self._pipeline is not None:
            try:
                cycle_result = self._pipeline.run_cycle(problem, pipeline_context)
                reasoning_summary = cycle_result.reasoning_summary
                validation_status = cycle_result.validation_status
                critique_summary = getattr(cycle_result, "self_critic_summary", "No critique available.")
                pipeline_confidence = cycle_result.metadata.get("confidence_score", 0.75)
            except Exception as exc:
                logger.error("Pipeline error: %s", exc, exc_info=True)
                reasoning_summary = f"Autonomous reasoning encountered an error: {exc}"
                validation_status = "error"
                critique_summary = "Analysis incomplete."
                pipeline_confidence = 0.5
        else:
            reasoning_summary = (
                f"Framework-only analysis for: {problem[:120]}\n"
                "Autonomous reasoning pipeline is offline (model download unavailable). "
                "Structured framework analyses are provided below."
            )
            validation_status = "framework_only"
            critique_summary = (
                "Full self-critique requires the autonomous pipeline. "
                "Review the framework analyses for actionable recommendations."
            )
            pipeline_confidence = 0.75

        # 3. Framework analyses
        from .aic_system import BusinessContext, ConsultingFramework
        biz_context = BusinessContext(
            industry=context.get("industry", "general"),
            company_size=context.get("company_size", "sme"),
            market_position=context.get("market_position", "challenger"),
            strategic_priorities=context.get("strategic_priorities", []),
            cultural_context=context.get("cultural_context", ""),
        )

        recommended_frameworks = self._framework_engine.recommend_frameworks(biz_context)
        try:
            framework_analyses = self._framework_engine.analyse(
                problem, biz_context, recommended_frameworks
            )
        except Exception as exc:
            logger.error("Framework analysis error: %s", exc)
            framework_analyses = {"error": str(exc)}

        # Critique score from metadata or default
        critique_score = float(
            cycle_result.metadata.get("critique_score", 0.7)
            if cycle_result is not None
            else 0.7
        )

        report = ConsultingReport(
            session_id=session_id,
            problem=problem,
            reasoning_summary=reasoning_summary,
            validation_status=validation_status,
            critique_summary=critique_summary,
            critique_score=critique_score,
            framework_analyses=framework_analyses,
            recommended_frameworks=recommended_frameworks,
            confidence=pipeline_confidence,
            context=context,
            created_at=datetime.now(),
            past_sessions_used=len(past_sessions),
            metadata={
                "memory_dir": self._memory_dir,
                "frameworks_applied": [f.value for f in recommended_frameworks],
            },
        )

        # 4. Persist
        if self._session_repo:
            try:
                self._session_repo.save_session(report.to_dict())
            except Exception as exc:
                logger.warning("Could not save session: %s", exc)

        return report

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_init()
        if self._session_repo:
            return self._session_repo.get_session(session_id)
        return None

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        self._ensure_init()
        if self._session_repo:
            return self._session_repo.list_sessions(limit=limit)
        return []
