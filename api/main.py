"""AIC REST API — FastAPI application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_consulting_service
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    SessionSummary,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up ConsultingService…")
    get_consulting_service()
    logger.info("ConsultingService ready.")
    yield


app = FastAPI(
    title="AIC — AI Consulting System",
    description="REST API for the autonomous consulting analysis pipeline.",
    version="5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse(status="ok")


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["consulting"])
def analyze(request: AnalyzeRequest, svc=Depends(get_consulting_service)):
    """Run a full consulting analysis and return a structured report."""
    context = request.context.model_dump() if request.context else {}
    try:
        report = svc.analyze(request.problem, context)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Analysis error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed. See server logs.")

    return AnalyzeResponse(
        session_id=report.session_id,
        problem=report.problem,
        reasoning_summary=report.reasoning_summary,
        validation_status=report.validation_status,
        critique_summary=report.critique_summary,
        critique_score=report.critique_score,
        framework_analyses=report.framework_analyses,
        recommended_frameworks=[str(f) for f in report.recommended_frameworks],
        confidence=report.confidence,
        past_sessions_used=report.past_sessions_used,
        created_at=report.created_at.isoformat(),
    )


@app.get("/api/v1/sessions", response_model=List[SessionSummary], tags=["sessions"])
def list_sessions(limit: int = 20, svc=Depends(get_consulting_service)):
    sessions = svc.list_sessions(limit=limit)
    return [
        SessionSummary(
            session_id=s["session_id"],
            problem=s.get("problem", ""),
            confidence=s.get("confidence", 0.0),
            validation_status=s.get("validation_status", "unknown"),
            created_at=s.get("created_at", ""),
        )
        for s in sessions
    ]


@app.get("/api/v1/sessions/{session_id}", response_model=AnalyzeResponse, tags=["sessions"])
def get_session(session_id: str, svc=Depends(get_consulting_service)):
    session = svc.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return AnalyzeResponse(
        session_id=session["session_id"],
        problem=session.get("problem", ""),
        reasoning_summary=session.get("reasoning_summary", ""),
        validation_status=session.get("validation_status", "unknown"),
        critique_summary=session.get("critique_summary", ""),
        critique_score=session.get("critique_score", 0.0),
        framework_analyses=session.get("framework_analyses", {}),
        recommended_frameworks=session.get("recommended_frameworks", []),
        confidence=session.get("confidence", 0.0),
        past_sessions_used=session.get("past_sessions_used", 0),
        created_at=session.get("created_at", ""),
    )


@app.get("/api/v1/memory/stats", tags=["system"])
def memory_stats(svc=Depends(get_consulting_service)):
    sessions = svc.list_sessions(limit=1000)
    return {
        "total_sessions": len(sessions),
        "validated": sum(1 for s in sessions if s.get("validation_status") == "valid"),
        "avg_confidence": (
            round(sum(s.get("confidence", 0) for s in sessions) / len(sessions), 4)
            if sessions else 0.0
        ),
    }
