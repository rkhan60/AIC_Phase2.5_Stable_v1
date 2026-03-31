"""logic_engine — backward-compatibility shim.

The original file contained ~1 767 lines of untrained PyTorch neural-network
classes (AICConsultingModel, ReasoningEngine, ConsultingMemorySystem, etc.)
that were never called with real weights and produced no meaningful output.

This shim preserves every public name that was imported elsewhere so the rest
of the codebase continues to work without changes:

    from core.engine.logic_engine import AICConsultingModel, create_aic_system
    from core.engine.logic_engine import ConsultingRole, ReasoningType
"""

from .enums import ConsultingRole, ReasoningType  # noqa: F401  (re-exported)

from ..aic_system import (  # noqa: F401  (re-exported)
    BusinessIntelligenceType,
    ConsultingFramework,
    BusinessContext,
    ConsultingFrameworkEngine,
    create_aic_system,
)

# Backward-compat alias: callers that do `AICConsultingModel()` now get a
# ConsultingFrameworkEngine instance instead of crashing on missing PyTorch
# weights.
AICConsultingModel = ConsultingFrameworkEngine

__all__ = [
    "ConsultingRole",
    "ReasoningType",
    "BusinessIntelligenceType",
    "ConsultingFramework",
    "BusinessContext",
    "ConsultingFrameworkEngine",
    "AICConsultingModel",
    "create_aic_system",
]
