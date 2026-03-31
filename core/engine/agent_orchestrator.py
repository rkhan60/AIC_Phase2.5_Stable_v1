"""Agent orchestrator — routes analysis requests through ConsultingService."""

from pathlib import Path
from typing import Any, Dict, Optional


def run_agentic_analysis(
    user_question: str,
    reasoning_path: str = "deductive",
    memory_config: Optional[Dict] = None,
    industry: str = "general",
    role: str = "consultant",
    response_type: str = "Consulting Report",
) -> Dict[str, Any]:
    """Run a full consulting analysis via ConsultingService.

    Parameters mirror the original simulation signature so existing callers
    continue to work without modification.
    """
    base_dir = Path(__file__).parent.parent.parent
    db_path = str(base_dir / "data" / "aic.db")
    (base_dir / "data").mkdir(parents=True, exist_ok=True)
    (base_dir / "memory").mkdir(exist_ok=True)

    from ..consulting_service import ConsultingService

    svc = ConsultingService(
        memory_dir=str(base_dir / "memory"),
        db_path=db_path,
    )

    context: Dict[str, Any] = {
        "industry": industry,
        "company_size": (memory_config or {}).get("company_size", "sme"),
        "market_position": (memory_config or {}).get("market_position", "challenger"),
        "reasoning_path": reasoning_path,
        "role": role,
        "response_type": response_type,
    }

    report = svc.analyze(user_question, context)

    return {
        "session_id": report.session_id,
        "reasoning_weights": [{"deductive": 0.8, "inductive": 0.6}],
        "analysis": report.reasoning_summary,
        "framework_analyses": report.framework_analyses,
        "recommended_frameworks": [str(f) for f in report.recommended_frameworks],
        "critique": report.critique_summary,
        "memory_confidence": report.confidence,
        "pattern_match": report.confidence,
        "knowledge_score": report.confidence,
        "learning_rate": 0.5,
        "confidence": report.confidence,
        "past_sessions_used": report.past_sessions_used,
    }
