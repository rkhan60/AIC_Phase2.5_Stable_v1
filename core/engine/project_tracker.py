from typing import Dict, List, Optional, Union, Any
import asyncio
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import json
from pathlib import Path

from .base_agent import AgentAction, BusinessSummary

logger = logging.getLogger(__name__)

@dataclass
class ProjectMetadata:
    """Project metadata and configuration"""
    project_id: str
    name: str
    description: str
    start_date: datetime
    end_date: Optional[datetime]
    client: Dict[str, Any]
    objectives: List[str]
    success_criteria: List[str]
    constraints: Dict[str, Any]

@dataclass
class ProjectState:
    """Current state of the project"""
    status: str
    progress: float
    active_tasks: List[str]
    completed_tasks: List[str]
    blockers: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    last_updated: datetime

class ProjectTracker:
    """Central system for tracking and coordinating agent activities"""
    
    def __init__(self, project_metadata: ProjectMetadata, storage_path: Optional[Path] = None):
        self.metadata = project_metadata
        self.storage_path = storage_path or Path("project_logs")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize project state
        self.state = ProjectState(
            status="initialized",
            progress=0.0,
            active_tasks=[],
            completed_tasks=[],
            blockers=[],
            metrics={},
            last_updated=datetime.now()
        )
        
        # Initialize memory structures
        self.agent_actions: Dict[str, List[AgentAction]] = {}
        self.agent_insights: Dict[str, List[Dict[str, Any]]] = {}
        self.agent_artifacts: Dict[str, Dict[str, Any]] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
        # Setup logging
        self._setup_logging()
        
    async def record_agent_action(self, action: AgentAction):
        """Record an agent's action"""
        # Store action
        if action.agent_id not in self.agent_actions:
            self.agent_actions[action.agent_id] = []
        self.agent_actions[action.agent_id].append(action)
        
        # Update project state
        await self._update_project_state(action)
        
        # Log action
        await self._log_action(action)
        
        # Check for collaboration opportunities
        await self._check_collaboration_opportunities(action)
        
    async def record_insight(self, agent_id: str, insight: Dict[str, Any]):
        """Record an agent's insight"""
        if agent_id not in self.agent_insights:
            self.agent_insights[agent_id] = []
        self.agent_insights[agent_id].append({
            "timestamp": datetime.now(),
            "content": insight
        })
        
    async def record_artifact(self, agent_id: str, name: str, artifact: Any):
        """Record an agent's artifact"""
        if agent_id not in self.agent_artifacts:
            self.agent_artifacts[agent_id] = {}
        self.agent_artifacts[agent_id][name] = {
            "timestamp": datetime.now(),
            "content": artifact
        }
        
    async def get_project_summary(self) -> Dict[str, Any]:
        """Generate project summary"""
        return {
            "metadata": asdict(self.metadata),
            "state": asdict(self.state),
            "agent_statistics": self._generate_agent_statistics(),
            "key_insights": self._extract_key_insights(),
            "business_value": self._calculate_business_value()
        }
        
    async def get_agent_history(self, agent_id: str) -> Dict[str, Any]:
        """Get agent's history"""
        return {
            "actions": [asdict(action) for action in self.agent_actions.get(agent_id, [])],
            "insights": self.agent_insights.get(agent_id, []),
            "artifacts": self.agent_artifacts.get(agent_id, {})
        }
        
    async def _update_project_state(self, action: AgentAction):
        """Update project state based on agent action"""
        self.state.last_updated = datetime.now()
        
        if action.status == "completed":
            if action.action_id in self.state.active_tasks:
                self.state.active_tasks.remove(action.action_id)
                self.state.completed_tasks.append(action.action_id)
        elif action.status == "failed":
            self.state.blockers.append({
                "action_id": action.action_id,
                "agent_id": action.agent_id,
                "error": action.error,
                "timestamp": datetime.now()
            })
        else:
            if action.action_id not in self.state.active_tasks:
                self.state.active_tasks.append(action.action_id)
                
        # Update progress
        total_tasks = len(self.state.active_tasks) + len(self.state.completed_tasks)
        self.state.progress = len(self.state.completed_tasks) / total_tasks if total_tasks > 0 else 0.0
        
    async def _check_collaboration_opportunities(self, action: AgentAction):
        """Check for potential collaboration opportunities"""
        if action.status == "completed" and action.result:
            # Analyze action result for collaboration triggers
            collaboration_opportunities = self._analyze_collaboration_needs(action)
            
            for opportunity in collaboration_opportunities:
                self.collaboration_history.append({
                    "timestamp": datetime.now(),
                    "source_action": action.action_id,
                    "opportunity": opportunity
                })
                
    def _analyze_collaboration_needs(self, action: AgentAction) -> List[Dict[str, Any]]:
        """Analyze action for collaboration opportunities"""
        opportunities = []
        
        # Example collaboration triggers
        if "data_cleaning_needed" in action.result.get("technical_output", {}):
            opportunities.append({
                "type": "data_cleaning",
                "target_agent": "DataCleaner",
                "priority": "high"
            })
            
        if "visualization_needed" in action.result.get("technical_output", {}):
            opportunities.append({
                "type": "visualization",
                "target_agent": "EDAVisualizer",
                "priority": "medium"
            })
            
        return opportunities
        
    def _generate_agent_statistics(self) -> Dict[str, Any]:
        """Generate statistics about agent performance"""
        stats = {}
        for agent_id, actions in self.agent_actions.items():
            completed_actions = [a for a in actions if a.status == "completed"]
            failed_actions = [a for a in actions if a.status == "failed"]
            
            stats[agent_id] = {
                "total_actions": len(actions),
                "completed_actions": len(completed_actions),
                "failed_actions": len(failed_actions),
                "success_rate": len(completed_actions) / len(actions) if actions else 0,
                "average_duration": self._calculate_average_duration(completed_actions)
            }
            
        return stats
        
    def _extract_key_insights(self) -> List[Dict[str, Any]]:
        """Extract key insights from all agents"""
        all_insights = []
        for agent_id, insights in self.agent_insights.items():
            all_insights.extend([{
                "agent_id": agent_id,
                **insight
            } for insight in insights])
            
        # Sort by timestamp and take most recent/important insights
        return sorted(all_insights, key=lambda x: x["timestamp"], reverse=True)[:10]
        
    def _calculate_business_value(self) -> Dict[str, Any]:
        """Calculate overall business value delivered"""
        value_metrics = {
            "efficiency_gains": self._calculate_efficiency_gains(),
            "quality_improvements": self._calculate_quality_improvements(),
            "risk_reduction": self._calculate_risk_reduction(),
            "cost_savings": self._calculate_cost_savings()
        }
        
        return {
            "metrics": value_metrics,
            "summary": self._create_value_summary(value_metrics)
        }
        
    def _setup_logging(self):
        """Setup project logging"""
        log_file = self.storage_path / f"{self.metadata.project_id}_project.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
    async def _log_action(self, action: AgentAction):
        """Log action to storage"""
        action_log = self.storage_path / f"{self.metadata.project_id}_actions.jsonl"
        
        async with aiofiles.open(action_log, mode='a') as f:
            await f.write(json.dumps(asdict(action)) + '\n') 