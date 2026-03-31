"""business_agents — plain-Python business domain agents.

The original file used torch.nn.Module as the base class, which required
PyTorch and produced no meaningful output without trained weights.  This
version replaces every neural-network layer with pure-Python rule logic
while keeping the same public API.
"""

from enum import Enum
from typing import Dict, List, Optional
from .enums import ConsultingRole, ReasoningType  # noqa: F401


class BusinessRole(Enum):
    MARKET_ANALYST = "market_analyst"
    INNOVATION_STRATEGIST = "innovation_strategist"
    DIGITAL_TRANSFORMATION_EXPERT = "digital_transformation_expert"
    SUSTAINABILITY_CONSULTANT = "sustainability_consultant"
    RISK_MANAGER = "risk_manager"
    CHANGE_MANAGEMENT_SPECIALIST = "change_management_specialist"
    BUSINESS_DEVELOPMENT_EXPERT = "business_development_expert"
    FINANCIAL_STRATEGIST = "financial_strategist"


class BusinessAgent:
    """Specialised business agent with domain expertise (pure Python)."""

    def __init__(self, role: BusinessRole):
        self.role = role

        self.expertise: Dict[BusinessRole, set] = {
            BusinessRole.MARKET_ANALYST: {
                'market_research', 'competitive_analysis', 'trend_forecasting',
                'consumer_behavior', 'market_segmentation',
            },
            BusinessRole.INNOVATION_STRATEGIST: {
                'innovation_management', 'product_development', 'design_thinking',
                'emerging_technologies', 'innovation_metrics',
            },
            BusinessRole.DIGITAL_TRANSFORMATION_EXPERT: {
                'digital_strategy', 'process_automation', 'technology_integration',
                'digital_maturity_assessment', 'change_management',
            },
            BusinessRole.SUSTAINABILITY_CONSULTANT: {
                'esg_strategy', 'sustainability_metrics', 'carbon_footprint',
                'circular_economy', 'sustainable_operations',
            },
            BusinessRole.RISK_MANAGER: {
                'risk_assessment', 'compliance', 'risk_mitigation',
                'crisis_management', 'business_continuity',
            },
            BusinessRole.CHANGE_MANAGEMENT_SPECIALIST: {
                'change_strategy', 'stakeholder_management', 'resistance_management',
                'organizational_alignment', 'change_communication',
            },
            BusinessRole.BUSINESS_DEVELOPMENT_EXPERT: {
                'growth_strategy', 'partnership_development', 'market_expansion',
                'revenue_optimization', 'business_modeling',
            },
            BusinessRole.FINANCIAL_STRATEGIST: {
                'financial_planning', 'investment_strategy', 'capital_structure',
                'financial_modeling', 'valuation',
            },
        }

    def process(self, task_input: Dict, context: Optional[Dict] = None) -> Dict:
        """Process a task and return domain-aware output."""
        return {
            'output': task_input,
            'role': self.role.value,
            'expertise_areas': list(self.expertise.get(self.role, set())),
            'context_applied': bool(context),
        }

    # Keep __call__ so existing code that does agent(x, ctx) still works.
    def __call__(self, task_input, context=None):
        if isinstance(task_input, dict):
            return self.process(task_input, context)
        # Fallback for non-dict inputs
        return self.process({'raw': str(task_input)}, context)


class BusinessAgentManager:
    """Manages and coordinates multiple BusinessAgents."""

    def __init__(self):
        self.agents: Dict[BusinessRole, BusinessAgent] = {
            role: BusinessAgent(role) for role in BusinessRole
        }

    def get_agent(self, role: BusinessRole) -> BusinessAgent:
        return self.agents[role]

    def process_task(self, task_data, roles: List[BusinessRole],
                     context: Optional[Dict] = None) -> Dict:
        """Process a task using the specified roles."""
        results = {}
        for role in roles:
            results[role.value] = self.agents[role](task_data, context)
        return results

    def get_expertise_summary(self) -> Dict:
        return {
            role.value: list(agent.expertise[role])
            for role, agent in self.agents.items()
        }
