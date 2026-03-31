"""business_agents.py — plain-Python business agent layer (no PyTorch)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from .enums import ConsultingRole, ReasoningType


class BusinessRole(Enum):
    MARKET_ANALYST = "market_analyst"
    INNOVATION_STRATEGIST = "innovation_strategist"
    DIGITAL_TRANSFORMATION_EXPERT = "digital_transformation_expert"
    SUSTAINABILITY_CONSULTANT = "sustainability_consultant"
    RISK_MANAGER = "risk_manager"
    CHANGE_MANAGEMENT_SPECIALIST = "change_management_specialist"
    BUSINESS_DEVELOPMENT_EXPERT = "business_development_expert"
    FINANCIAL_STRATEGIST = "financial_strategist"


_EXPERTISE: Dict[BusinessRole, set] = {
    BusinessRole.MARKET_ANALYST: {
        "market_research", "competitive_analysis", "trend_forecasting",
        "consumer_behavior", "market_segmentation",
    },
    BusinessRole.INNOVATION_STRATEGIST: {
        "innovation_management", "product_development", "design_thinking",
        "emerging_technologies", "innovation_metrics",
    },
    BusinessRole.DIGITAL_TRANSFORMATION_EXPERT: {
        "digital_strategy", "process_automation", "technology_integration",
        "digital_maturity_assessment", "change_management",
    },
    BusinessRole.SUSTAINABILITY_CONSULTANT: {
        "esg_strategy", "sustainability_metrics", "carbon_footprint",
        "circular_economy", "sustainable_operations",
    },
    BusinessRole.RISK_MANAGER: {
        "risk_assessment", "compliance", "risk_mitigation",
        "crisis_management", "business_continuity",
    },
    BusinessRole.CHANGE_MANAGEMENT_SPECIALIST: {
        "change_strategy", "stakeholder_management", "resistance_management",
        "organizational_alignment", "change_communication",
    },
    BusinessRole.BUSINESS_DEVELOPMENT_EXPERT: {
        "growth_strategy", "partnership_development", "market_expansion",
        "revenue_optimization", "business_modeling",
    },
    BusinessRole.FINANCIAL_STRATEGIST: {
        "financial_planning", "investment_strategy", "capital_structure",
        "financial_modeling", "valuation",
    },
}


class BusinessAgent:
    """Lightweight domain expert — no neural network required."""

    def __init__(self, role: BusinessRole):
        self.role = role
        self.expertise: set = _EXPERTISE.get(role, set())

    def process(
        self,
        task_input: Dict[str, Any],
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        return {
            "output": task_input,
            "role": self.role.value,
            "expertise_areas": list(self.expertise),
        }

    def __call__(
        self,
        task_input: Dict[str, Any],
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        return self.process(task_input, context)


class BusinessAgentManager:
    """Registry of one BusinessAgent per role."""

    def __init__(self):
        self.agents: Dict[BusinessRole, BusinessAgent] = {
            role: BusinessAgent(role) for role in BusinessRole
        }

    def get_agent(self, role: BusinessRole) -> BusinessAgent:
        return self.agents[role]
