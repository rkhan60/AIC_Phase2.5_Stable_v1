from .enums import ConsultingRole, ReasoningType
from .logic_engine import (
    AICConsultingModel,          # backward-compat alias for ConsultingFrameworkEngine
    ConsultingFrameworkEngine,
    BusinessContext,
    create_aic_system,
)
from .business_agents import (
    BusinessRole,
    BusinessAgent,
    BusinessAgentManager,
)
from .agent_diagnostics import AgentDiagnostics
from .parallel_processor import (
    ParallelAgentProcessor,
    AgentGroup,
)

__all__ = [
    'AICConsultingModel',
    'ConsultingFrameworkEngine',
    'BusinessContext',
    'ConsultingRole',
    'ReasoningType',
    'create_aic_system',
    'BusinessRole',
    'BusinessAgent',
    'BusinessAgentManager',
    'AgentDiagnostics',
    'ParallelAgentProcessor',
    'AgentGroup',
]
