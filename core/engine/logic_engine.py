"""logic_engine.py — thin compatibility shim.

All real implementation lives in core/aic_system.py and core/engine/enums.py.
This module re-exports everything so existing import paths keep working.
"""
from .enums import ConsultingRole, ReasoningType
from ..aic_system import (
    BusinessIntelligenceType,
    ConsultingFramework,
    BusinessContext,
    ConsultingFrameworkEngine,
)

# Backward-compat alias used in a few places
AICConsultingModel = ConsultingFrameworkEngine


def create_aic_system() -> ConsultingFrameworkEngine:
    """Return a fresh ConsultingFrameworkEngine instance."""
    return ConsultingFrameworkEngine()
