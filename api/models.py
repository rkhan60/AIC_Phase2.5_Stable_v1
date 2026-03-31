"""Pydantic request/response models for the AIC REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BusinessContextRequest(BaseModel):
    industry: str = "general"
    company_size: str = "sme"
    market_position: str = "challenger"
    strategic_priorities: List[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    problem: str = Field(..., min_length=1, description="Business problem statement")
    context: Optional[BusinessContextRequest] = None


class AnalyzeResponse(BaseModel):
    session_id: str
    problem: str
    reasoning_summary: str
    validation_status: str
    critique_summary: str
    critique_score: float
    framework_analyses: Dict[str, Any]
    recommended_frameworks: List[Any]
    confidence: float
    past_sessions_used: int
    created_at: str


class SessionSummary(BaseModel):
    session_id: str
    problem: str
    confidence: float
    validation_status: str
    created_at: str


class HealthResponse(BaseModel):
    status: str
    version: str = "5.0"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
