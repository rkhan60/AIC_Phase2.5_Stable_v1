from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging
from ..memory.memory_system import MemoryItem, MemoryType, EmotionalTag
from ..memory.memory_query import MemoryQueryEngine

logger = logging.getLogger(__name__)

class GoalType(Enum):
    STRATEGIC = "strategic"  # Long-term strategic objectives
    TACTICAL = "tactical"    # Medium-term tactical goals
    IMMEDIATE = "immediate" # Short-term immediate actions
    ADAPTIVE = "adaptive"   # Goals that adapt to changing conditions

class GoalStatus(Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"

@dataclass
class GoalMetrics:
    """Metrics for goal evaluation"""
    clarity: float = 0.0      # How clear and well-defined
    feasibility: float = 0.0  # How achievable
    impact: float = 0.0       # Expected impact
    urgency: float = 0.0      # Time sensitivity
    alignment: float = 0.0    # Alignment with higher goals
    
    def get_priority_score(self) -> float:
        """Calculate overall priority score"""
        weights = {
            'clarity': 0.2,
            'feasibility': 0.2,
            'impact': 0.3,
            'urgency': 0.2,
            'alignment': 0.1
        }
        return sum(
            getattr(self, metric) * weight 
            for metric, weight in weights.items()
        )

@dataclass
class Goal:
    """Represents a structured goal with reasoning"""
    id: str
    type: GoalType
    description: str
    success_criteria: List[str]
    dependencies: List[str] = field(default_factory=list)
    subgoals: List[str] = field(default_factory=list)
    metrics: GoalMetrics = field(default_factory=GoalMetrics)
    status: GoalStatus = GoalStatus.PROPOSED
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

class GoalPlanner:
    """Agent for translating input into structured goals"""
    def __init__(self, memory_query: MemoryQueryEngine):
        self.memory_query = memory_query
        self.goals: Dict[str, Goal] = {}
        self.reasoning_history: List[Dict[str, Any]] = []
        
    def create_goal(self, 
                   input_data: Any,
                   context: Dict[str, Any]) -> Goal:
        """Create structured goal from input"""
        # Generate goal ID
        goal_id = self._generate_goal_id(input_data)
        
        # Analyze input and context
        goal_type = self._determine_goal_type(input_data, context)
        success_criteria = self._define_success_criteria(input_data, context)
        
        # Create initial goal
        goal = Goal(
            id=goal_id,
            type=goal_type,
            description=str(input_data),
            success_criteria=success_criteria
        )
        
        # Evaluate and set metrics
        goal.metrics = self._evaluate_goal_metrics(goal, context)
        
        # Document reasoning
        reasoning = {
            'timestamp': datetime.now(),
            'input': input_data,
            'context': context,
            'analysis': {
                'goal_type': goal_type.value,
                'metrics': goal.metrics.__dict__,
                'rationale': self._generate_rationale(goal)
            }
        }
        goal.reasoning_trace.append(reasoning)
        self.reasoning_history.append(reasoning)
        
        # Store goal
        self.goals[goal_id] = goal
        return goal
        
    def update_goal(self,
                   goal_id: str,
                   updates: Dict[str, Any]) -> Optional[Goal]:
        """Update existing goal"""
        if goal_id not in self.goals:
            return None
            
        goal = self.goals[goal_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(goal, key):
                setattr(goal, key, value)
                
        goal.modified_at = datetime.now()
        
        # Document update reasoning
        reasoning = {
            'timestamp': datetime.now(),
            'update_type': 'goal_update',
            'changes': updates,
            'rationale': self._generate_rationale(goal)
        }
        goal.reasoning_trace.append(reasoning)
        
        return goal
        
    def decompose_goal(self, goal_id: str) -> List[Goal]:
        """Break down goal into subgoals"""
        if goal_id not in self.goals:
            return []
            
        parent_goal = self.goals[goal_id]
        subgoals = []
        
        # Query similar past goals
        similar_goals = self.memory_query.find_similar_experiences(
            parent_goal.description,
            [MemoryItem(
                content=g.description,
                memory_type=MemoryType.WORKING,
                context_tags=[g.type.value],
                emotional_tags=[EmotionalTag.NEUTRAL],
                importance_score=g.metrics.get_priority_score(),
                last_accessed=g.modified_at
            ) for g in self.goals.values()]
        )
        
        # Create subgoals based on success criteria
        for i, criterion in enumerate(parent_goal.success_criteria):
            subgoal_id = f"{goal_id}_sub_{i}"
            subgoal = Goal(
                id=subgoal_id,
                type=GoalType.TACTICAL,
                description=f"Achieve: {criterion}",
                success_criteria=[criterion],
                dependencies=[goal_id]
            )
            
            # Set metrics based on parent and similar goals
            subgoal.metrics = self._evaluate_goal_metrics(
                subgoal,
                {'parent_goal': parent_goal, 'similar_goals': similar_goals}
            )
            
            subgoals.append(subgoal)
            self.goals[subgoal_id] = subgoal
            parent_goal.subgoals.append(subgoal_id)
            
        return subgoals
        
    def _generate_goal_id(self, input_data: Any) -> str:
        """Generate unique goal ID"""
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{str(input_data)}"
        return f"goal_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"
        
    def _determine_goal_type(self,
                           input_data: Any,
                           context: Dict[str, Any]) -> GoalType:
        """Determine appropriate goal type"""
        # Implementation would analyze input and context
        return GoalType.STRATEGIC
        
    def _define_success_criteria(self,
                               input_data: Any,
                               context: Dict[str, Any]) -> List[str]:
        """Define measurable success criteria"""
        # Implementation would extract concrete criteria
        return ["Criterion 1", "Criterion 2"]
        
    def _evaluate_goal_metrics(self,
                             goal: Goal,
                             context: Dict[str, Any]) -> GoalMetrics:
        """Evaluate goal metrics"""
        metrics = GoalMetrics()
        
        # Evaluate clarity
        metrics.clarity = self._evaluate_clarity(goal)
        
        # Evaluate feasibility
        metrics.feasibility = self._evaluate_feasibility(goal, context)
        
        # Evaluate impact
        metrics.impact = self._evaluate_impact(goal, context)
        
        # Evaluate urgency
        metrics.urgency = self._evaluate_urgency(goal, context)
        
        # Evaluate alignment
        metrics.alignment = self._evaluate_alignment(goal, context)
        
        return metrics
        
    def _evaluate_clarity(self, goal: Goal) -> float:
        """Evaluate goal clarity"""
        factors = [
            bool(goal.description),
            bool(goal.success_criteria),
            len(goal.success_criteria) > 0,
            all(len(c) > 10 for c in goal.success_criteria)  # Minimum detail
        ]
        return sum(1 for f in factors if f) / len(factors)
        
    def _evaluate_feasibility(self, goal: Goal, context: Dict[str, Any]) -> float:
        """Evaluate goal feasibility"""
        # Implementation would assess resources, constraints, etc.
        return 0.7
        
    def _evaluate_impact(self, goal: Goal, context: Dict[str, Any]) -> float:
        """Evaluate expected impact"""
        # Implementation would assess potential outcomes
        return 0.8
        
    def _evaluate_urgency(self, goal: Goal, context: Dict[str, Any]) -> float:
        """Evaluate time sensitivity"""
        # Implementation would assess temporal factors
        return 0.5
        
    def _evaluate_alignment(self, goal: Goal, context: Dict[str, Any]) -> float:
        """Evaluate alignment with higher goals"""
        # Implementation would assess goal hierarchy
        return 0.9
        
    def _generate_rationale(self, goal: Goal) -> str:
        """Generate reasoning rationale"""
        return f"Goal evaluated with clarity={goal.metrics.clarity}, " \
               f"feasibility={goal.metrics.feasibility}, " \
               f"impact={goal.metrics.impact}"
        
    def get_goal_trace(self, goal_id: str) -> List[Dict[str, Any]]:
        """Get complete reasoning trace for goal"""
        if goal_id not in self.goals:
            return []
        return self.goals[goal_id].reasoning_trace 