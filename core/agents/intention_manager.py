from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import logging
from .goal_planner import Goal, GoalStatus, GoalType
from ..memory.memory_query import MemoryQueryEngine

logger = logging.getLogger(__name__)

class IntentionStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class IntentionMetrics:
    """Metrics for intention evaluation"""
    priority: float = 0.0     # Overall priority
    progress: float = 0.0     # Progress towards completion
    resources: float = 0.0    # Resource availability
    conflicts: float = 0.0    # Conflict with other intentions
    momentum: float = 0.0     # Recent progress rate

@dataclass
class Intention:
    """Represents an active intention with execution state"""
    id: str
    goal_id: str
    status: IntentionStatus
    metrics: IntentionMetrics
    dependencies: Set[str] = field(default_factory=set)
    blockers: Set[str] = field(default_factory=set)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

class IntentionManager:
    """Agent for managing active intentions and their execution"""
    def __init__(self, memory_query: MemoryQueryEngine):
        self.memory_query = memory_query
        self.intentions: Dict[str, Intention] = {}
        self.active_goals: Dict[str, Goal] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    def create_intention(self,
                        goal: Goal,
                        context: Dict[str, Any]) -> Intention:
        """Create intention from goal"""
        # Generate intention ID
        intention_id = self._generate_intention_id(goal)
        
        # Evaluate initial metrics
        metrics = self._evaluate_intention_metrics(goal, context)
        
        # Create intention
        intention = Intention(
            id=intention_id,
            goal_id=goal.id,
            status=IntentionStatus.ACTIVE,
            metrics=metrics
        )
        
        # Set scheduling
        schedule = self._schedule_intention(intention, context)
        intention.scheduled_start = schedule['start']
        intention.scheduled_end = schedule['end']
        
        # Document creation reasoning
        reasoning = {
            'timestamp': datetime.now(),
            'action': 'create_intention',
            'goal': goal.__dict__,
            'metrics': metrics.__dict__,
            'schedule': {
                'start': intention.scheduled_start,
                'end': intention.scheduled_end
            },
            'rationale': self._generate_rationale(intention)
        }
        intention.execution_trace.append(reasoning)
        self.execution_history.append(reasoning)
        
        # Store intention and goal
        self.intentions[intention_id] = intention
        self.active_goals[goal.id] = goal
        
        return intention
        
    def update_intention(self,
                        intention_id: str,
                        updates: Dict[str, Any]) -> Optional[Intention]:
        """Update intention state"""
        if intention_id not in self.intentions:
            return None
            
        intention = self.intentions[intention_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(intention, key):
                setattr(intention, key, value)
                
        intention.modified_at = datetime.now()
        
        # Document update
        reasoning = {
            'timestamp': datetime.now(),
            'action': 'update_intention',
            'updates': updates,
            'rationale': self._generate_rationale(intention)
        }
        intention.execution_trace.append(reasoning)
        
        return intention
        
    def evaluate_intentions(self) -> List[Dict[str, Any]]:
        """Evaluate all active intentions"""
        evaluations = []
        
        for intention in self.intentions.values():
            if intention.status != IntentionStatus.ACTIVE:
                continue
                
            # Get goal
            goal = self.active_goals.get(intention.goal_id)
            if not goal:
                continue
                
            # Evaluate progress
            progress = self._evaluate_progress(intention, goal)
            
            # Check for completion
            if progress >= 1.0:
                self.complete_intention(intention.id)
                
            # Check for blockers
            blockers = self._check_blockers(intention)
            if blockers:
                self.suspend_intention(intention.id, blockers)
                
            evaluations.append({
                'intention_id': intention.id,
                'goal_id': goal.id,
                'progress': progress,
                'blockers': blockers,
                'metrics': intention.metrics.__dict__
            })
            
        return evaluations
        
    def complete_intention(self, intention_id: str):
        """Mark intention as completed"""
        intention = self.intentions.get(intention_id)
        if not intention:
            return
            
        intention.status = IntentionStatus.COMPLETED
        intention.modified_at = datetime.now()
        
        # Update goal status
        goal = self.active_goals.get(intention.goal_id)
        if goal:
            goal.status = GoalStatus.COMPLETED
            
        # Document completion
        reasoning = {
            'timestamp': datetime.now(),
            'action': 'complete_intention',
            'metrics': intention.metrics.__dict__,
            'rationale': "Intention completed successfully"
        }
        intention.execution_trace.append(reasoning)
        
    def suspend_intention(self, intention_id: str, blockers: Set[str]):
        """Suspend intention due to blockers"""
        intention = self.intentions.get(intention_id)
        if not intention:
            return
            
        intention.status = IntentionStatus.SUSPENDED
        intention.blockers = blockers
        intention.modified_at = datetime.now()
        
        # Document suspension
        reasoning = {
            'timestamp': datetime.now(),
            'action': 'suspend_intention',
            'blockers': list(blockers),
            'rationale': "Intention suspended due to blockers"
        }
        intention.execution_trace.append(reasoning)
        
    def _generate_intention_id(self, goal: Goal) -> str:
        """Generate unique intention ID"""
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{goal.id}"
        return f"intention_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"
        
    def _evaluate_intention_metrics(self,
                                  goal: Goal,
                                  context: Dict[str, Any]) -> IntentionMetrics:
        """Evaluate intention metrics"""
        metrics = IntentionMetrics()
        
        # Set priority based on goal metrics
        metrics.priority = goal.metrics.get_priority_score()
        
        # Evaluate resource availability
        metrics.resources = self._evaluate_resources(goal, context)
        
        # Check for conflicts
        metrics.conflicts = self._evaluate_conflicts(goal)
        
        # Initialize progress and momentum
        metrics.progress = 0.0
        metrics.momentum = 1.0
        
        return metrics
        
    def _evaluate_resources(self, goal: Goal, context: Dict[str, Any]) -> float:
        """Evaluate resource availability"""
        # Implementation would check actual resource availability
        return 0.8
        
    def _evaluate_conflicts(self, goal: Goal) -> float:
        """Evaluate conflicts with other intentions"""
        conflict_score = 0.0
        total_intentions = len(self.intentions)
        
        if total_intentions == 0:
            return conflict_score
            
        for intention in self.intentions.values():
            if intention.status != IntentionStatus.ACTIVE:
                continue
                
            other_goal = self.active_goals.get(intention.goal_id)
            if not other_goal:
                continue
                
            # Check for resource conflicts
            if goal.type == other_goal.type:
                conflict_score += 0.2
                
            # Check for dependency conflicts
            if goal.id in intention.dependencies:
                conflict_score += 0.5
                
        return min(1.0, conflict_score)
        
    def _schedule_intention(self,
                          intention: Intention,
                          context: Dict[str, Any]) -> Dict[str, datetime]:
        """Schedule intention execution"""
        now = datetime.now()
        
        # Simple scheduling - implementation would be more sophisticated
        return {
            'start': now,
            'end': now + timedelta(days=7)  # Default 1 week duration
        }
        
    def _evaluate_progress(self, intention: Intention, goal: Goal) -> float:
        """Evaluate intention progress"""
        if not goal.success_criteria:
            return 0.0
            
        # Count met criteria
        met_criteria = sum(1 for _ in goal.success_criteria)  # Would check each criterion
        return met_criteria / len(goal.success_criteria)
        
    def _check_blockers(self, intention: Intention) -> Set[str]:
        """Check for blocking conditions"""
        blockers = set()
        
        # Check dependencies
        for dep_id in intention.dependencies:
            dep_intention = self.intentions.get(dep_id)
            if dep_intention and dep_intention.status != IntentionStatus.COMPLETED:
                blockers.add(f"dependency_{dep_id}")
                
        # Check resources
        if intention.metrics.resources < 0.2:
            blockers.add("insufficient_resources")
            
        # Check conflicts
        if intention.metrics.conflicts > 0.8:
            blockers.add("high_conflicts")
            
        return blockers
        
    def _generate_rationale(self, intention: Intention) -> str:
        """Generate reasoning rationale"""
        return f"Intention evaluated with priority={intention.metrics.priority}, " \
               f"progress={intention.metrics.progress}, " \
               f"resources={intention.metrics.resources}"
        
    def get_intention_trace(self, intention_id: str) -> List[Dict[str, Any]]:
        """Get complete execution trace for intention"""
        if intention_id not in self.intentions:
            return []
        return self.intentions[intention_id].execution_trace

    # ------------------------------------------------------------------
    # Pipeline-facing API (used by AutonomousPipeline)
    # ------------------------------------------------------------------

    def process_goals(self, hierarchy: Any, context: Dict[str, Any]) -> Intention:
        """Create and return the primary intention from a GoalHierarchy.

        Args:
            hierarchy: A GoalHierarchy whose main_goal drives the intention.
            context: Execution context passed to create_intention().

        Returns:
            An active Intention for the main goal.
        """
        return self.create_intention(hierarchy.main_goal, context)

    def select_intention(
        self, goals: List[Goal], context: Dict[str, Any]
    ) -> Optional[Intention]:
        """Select the highest-priority goal from a list and create an intention.

        Args:
            goals: Candidate Goal objects to choose from.
            context: Execution context passed to create_intention().

        Returns:
            An Intention for the best goal, or None if the list is empty.
        """
        if not goals:
            return None
        best_goal = max(goals, key=lambda g: g.metrics.get_priority_score())
        return self.create_intention(best_goal, context)