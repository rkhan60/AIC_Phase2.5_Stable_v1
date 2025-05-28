from enum import Enum
import torch
import torch.nn as nn
from typing import Dict, List, Optional
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

class BusinessAgent(nn.Module):
    """Specialized business agent with domain expertise"""
    def __init__(self, role: BusinessRole, d_model: int):
        super().__init__()
        self.role = role
        self.d_model = d_model
        
        # Role-specific networks
        self.networks = {
            BusinessRole.MARKET_ANALYST: self._build_market_analysis_network(),
            BusinessRole.INNOVATION_STRATEGIST: self._build_innovation_network(),
            BusinessRole.DIGITAL_TRANSFORMATION_EXPERT: self._build_digital_transformation_network(),
            BusinessRole.SUSTAINABILITY_CONSULTANT: self._build_sustainability_network(),
            BusinessRole.RISK_MANAGER: self._build_risk_management_network(),
            BusinessRole.CHANGE_MANAGEMENT_SPECIALIST: self._build_change_management_network(),
            BusinessRole.BUSINESS_DEVELOPMENT_EXPERT: self._build_business_development_network(),
            BusinessRole.FINANCIAL_STRATEGIST: self._build_financial_strategy_network()
        }
        
        # Expertise areas for each role
        self.expertise = {
            BusinessRole.MARKET_ANALYST: {
                'market_research', 'competitive_analysis', 'trend_forecasting',
                'consumer_behavior', 'market_segmentation'
            },
            BusinessRole.INNOVATION_STRATEGIST: {
                'innovation_management', 'product_development', 'design_thinking',
                'emerging_technologies', 'innovation_metrics'
            },
            BusinessRole.DIGITAL_TRANSFORMATION_EXPERT: {
                'digital_strategy', 'process_automation', 'technology_integration',
                'digital_maturity_assessment', 'change_management'
            },
            BusinessRole.SUSTAINABILITY_CONSULTANT: {
                'esg_strategy', 'sustainability_metrics', 'carbon_footprint',
                'circular_economy', 'sustainable_operations'
            },
            BusinessRole.RISK_MANAGER: {
                'risk_assessment', 'compliance', 'risk_mitigation',
                'crisis_management', 'business_continuity'
            },
            BusinessRole.CHANGE_MANAGEMENT_SPECIALIST: {
                'change_strategy', 'stakeholder_management', 'resistance_management',
                'organizational_alignment', 'change_communication'
            },
            BusinessRole.BUSINESS_DEVELOPMENT_EXPERT: {
                'growth_strategy', 'partnership_development', 'market_expansion',
                'revenue_optimization', 'business_modeling'
            },
            BusinessRole.FINANCIAL_STRATEGIST: {
                'financial_planning', 'investment_strategy', 'capital_structure',
                'financial_modeling', 'valuation'
            }
        }
        
    def _build_market_analysis_network(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
    def _build_innovation_network(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
    def _build_digital_transformation_network(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
    def _build_sustainability_network(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
    def _build_risk_management_network(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
    def _build_change_management_network(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
    def _build_business_development_network(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
    def _build_financial_strategy_network(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
    def forward(self, x, context=None):
        # Process input through role-specific network
        processed = self.networks[self.role](x)
        
        # Add context if available
        if context is not None:
            processed = processed + context
            
        return {
            'output': processed,
            'role': self.role.value,
            'expertise_areas': list(self.expertise[self.role])
        }

class BusinessAgentManager:
    """Manages and coordinates multiple business agents"""
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.agents = {}
        
        # Initialize agents for each role
        for role in BusinessRole:
            self.agents[role] = BusinessAgent(role, d_model)
            
    def get_agent(self, role: BusinessRole) -> BusinessAgent:
        """Get a specific agent by role"""
        return self.agents[role]
        
    def process_task(self, task_data: torch.Tensor, roles: List[BusinessRole], context: Optional[Dict] = None):
        """Process task using multiple agents"""
        results = {}
        
        for role in roles:
            agent = self.agents[role]
            results[role.value] = agent(task_data, context)
            
        return results
        
    def get_expertise_summary(self) -> Dict:
        """Get summary of all available expertise"""
        summary = {}
        for role, agent in self.agents.items():
            summary[role.value] = list(agent.expertise[role])
        return summary 