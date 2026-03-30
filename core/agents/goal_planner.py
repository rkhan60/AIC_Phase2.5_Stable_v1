from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging
import uuid
from ..memory.memory_system import MemoryItem, MemoryType, EmotionalTag
from ..memory.memory_query import MemoryQueryEngine
from ..config import config

logger = logging.getLogger(__name__)

_MAX_DESCRIPTION_LENGTH = 2000

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

@dataclass
class GoalHierarchy:
    """Hierarchical structure of a main goal and its decomposed subgoals."""
    main_goal: Goal
    subgoals: List[Goal] = field(default_factory=list)
    depth: int = 1
    created_at: datetime = field(default_factory=datetime.now)


class GoalPlanner:
    """Agent for translating input into structured goals"""
    def __init__(self, memory_query: MemoryQueryEngine):
        self.memory_query = memory_query
        self.goals: Dict[str, Goal] = {}
        self.reasoning_history: List[Dict[str, Any]] = []
        
    def create_goal(self,
                   input_data: Any,
                   context: Dict[str, Any]) -> Goal:
        """Create structured goal from input.

        Args:
            input_data: Goal description (str) or any object convertible to str.
            context:    Execution context dict.  Non-dict values are silently
                        replaced with an empty dict.

        Raises:
            ValueError: If input_data is empty after conversion to str.
        """
        # --- Input validation ---
        if not input_data and input_data != 0:
            raise ValueError("input_data must not be empty")
        description = str(input_data).strip()
        if not description:
            raise ValueError("input_data must not be blank")
        if len(description) > _MAX_DESCRIPTION_LENGTH:
            logger.warning(
                "input_data truncated from %d to %d characters",
                len(description), _MAX_DESCRIPTION_LENGTH,
            )
            description = description[:_MAX_DESCRIPTION_LENGTH]
        if not isinstance(context, dict):
            logger.warning("context is not a dict (%s); using empty dict", type(context))
            context = {}

        # Generate goal ID
        goal_id = self._generate_goal_id(description)
        
        # Analyze input and context
        goal_type = self._determine_goal_type(description, context)
        success_criteria = self._define_success_criteria(description, context)
        
        # Create initial goal
        goal = Goal(
            id=goal_id,
            type=goal_type,
            description=description,
            success_criteria=success_criteria,
        )
        
        # Evaluate and set metrics
        goal.metrics = self._evaluate_goal_metrics(goal, context)
        
        # Document reasoning
        reasoning = {
            'timestamp': datetime.now(),
            'input': description,
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
        """Generate a collision-resistant unique goal ID using UUID4."""
        return f"goal_{uuid.uuid4().hex[:12]}"

    def _determine_goal_type(self,
                             input_data: Any,
                             context: Dict[str, Any]) -> GoalType:
        """Classify the goal type from keywords in the description."""
        desc = str(input_data).lower()
        cfg = config.goal

        if any(kw in desc for kw in cfg.immediate_keywords):
            return GoalType.IMMEDIATE
        if any(kw in desc for kw in cfg.adaptive_keywords):
            return GoalType.ADAPTIVE
        if any(kw in desc for kw in cfg.tactical_keywords) and not context.get('long_term'):
            return GoalType.TACTICAL
        return GoalType.STRATEGIC

    def _define_success_criteria(self,
                                 input_data: Any,
                                 context: Dict[str, Any]) -> List[str]:
        """Derive measurable success criteria from the description and context."""
        desc = str(input_data)
        criteria: List[str] = [f"Successfully address: {desc[:120]}",
                                "Measurable outcome documented and validated by stakeholders"]

        if context.get('required_accuracy'):
            criteria.append(f"Achieve target accuracy >= {context['required_accuracy']}")
        if context.get('time_constraint'):
            criteria.append(f"Complete within the defined time constraint: {context['time_constraint']}")
        if context.get('domain'):
            criteria.append(f"Solution validated for the {context['domain']} domain")
        if context.get('budget'):
            criteria.append(f"Delivered within budget: {context['budget']}")

        return criteria

    def _evaluate_goal_metrics(self,
                               goal: Goal,
                               context: Dict[str, Any]) -> GoalMetrics:
        """Compute all goal metrics."""
        return GoalMetrics(
            clarity=self._evaluate_clarity(goal),
            feasibility=self._evaluate_feasibility(goal, context),
            impact=self._evaluate_impact(goal, context),
            urgency=self._evaluate_urgency(goal, context),
            alignment=self._evaluate_alignment(goal, context),
        )

    def _evaluate_clarity(self, goal: Goal) -> float:
        """Score how clearly the goal is defined."""
        factors = [
            bool(goal.description),
            len(goal.description) >= 20,
            bool(goal.success_criteria),
            len(goal.success_criteria) >= 2,
            all(len(c) > 10 for c in goal.success_criteria),
        ]
        return round(sum(1 for f in factors if f) / len(factors), 4)

    def _evaluate_feasibility(self, goal: Goal, context: Dict[str, Any]) -> float:
        """Score how achievable the goal is, based on dependencies and context."""
        cfg = config.goal
        score = 0.50

        # Penalise for each hard dependency
        dep_penalty = min(cfg.max_dependency_penalty,
                          len(goal.dependencies) * cfg.dependency_penalty_per_item)
        score -= dep_penalty

        # Reward if resources or budget are mentioned
        if context.get('resources') or context.get('budget'):
            score += cfg.resource_availability_boost

        # Reward clear, detailed success criteria
        if len(goal.success_criteria) >= 3:
            score += cfg.clear_criteria_boost

        # Vague description is a negative signal
        if len(goal.description) < 20:
            score -= 0.10

        # A stated time constraint shows planning maturity
        if context.get('time_constraint') or context.get('deadline'):
            score += cfg.time_constraint_boost

        return round(max(0.0, min(1.0, score)), 4)

    def _evaluate_impact(self, goal: Goal, context: Dict[str, Any]) -> float:
        """Score the expected business impact of the goal."""
        cfg = config.goal
        score = 0.50
        desc_lower = goal.description.lower()

        impact_hits = sum(1 for kw in cfg.high_impact_keywords if kw in desc_lower)
        score += min(cfg.max_impact_keyword_boost,
                     impact_hits * cfg.impact_keyword_boost_per_hit)

        if goal.type == GoalType.STRATEGIC:
            score += cfg.strategic_type_impact_boost
        elif goal.type == GoalType.TACTICAL:
            score += cfg.tactical_type_impact_boost

        if len(goal.success_criteria) >= 3:
            score += 0.05

        return round(max(0.0, min(1.0, score)), 4)

    def _evaluate_urgency(self, goal: Goal, context: Dict[str, Any]) -> float:
        """Score the time-sensitivity of the goal."""
        cfg = config.goal
        score = 0.30
        desc_lower = goal.description.lower()

        if any(kw in desc_lower for kw in cfg.urgency_keywords):
            score += cfg.urgency_keyword_boost

        time_constraint = context.get('time_constraint')
        if time_constraint:
            try:
                if float(time_constraint) < 48:
                    score += cfg.time_constraint_tight_boost
                else:
                    score += cfg.time_constraint_boost
            except (TypeError, ValueError):
                score += cfg.time_constraint_boost

        if goal.type == GoalType.IMMEDIATE:
            score += cfg.immediate_goal_type_boost
        elif goal.type == GoalType.ADAPTIVE:
            score += cfg.adaptive_goal_type_boost

        return round(max(0.0, min(1.0, score)), 4)

    def _evaluate_alignment(self, goal: Goal, context: Dict[str, Any]) -> float:
        """Score how well the goal aligns with existing goals in the planner."""
        cfg = config.goal
        score = 0.60

        if context.get('strategic_priority') or context.get('domain'):
            score += cfg.alignment_context_boost

        existing = [g for g in self.goals.values() if g.id != goal.id]
        if not existing:
            return round(min(1.0, score), 4)

        desc_words = set(goal.description.lower().split())
        overlaps = []
        for other in existing:
            other_words = set(other.description.lower().split())
            if desc_words:
                overlaps.append(
                    len(desc_words & other_words) / len(desc_words)
                )
        if overlaps:
            score += (sum(overlaps) / len(overlaps)) * cfg.alignment_keyword_overlap_weight

        return round(max(0.0, min(1.0, score)), 4)
        
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

    def plan_goal(self, goal_input: str, context: Dict[str, Any]) -> GoalHierarchy:
        """Create a GoalHierarchy from a free-text goal description.

        Args:
            goal_input: Natural language description of the goal.
            context: Execution context (domain, constraints, etc.).

        Returns:
            GoalHierarchy with a main goal and zero or more subgoals.
        """
        main_goal = self.create_goal(goal_input, context)
        subgoals = self.decompose_goal(main_goal.id)
        return GoalHierarchy(
            main_goal=main_goal,
            subgoals=subgoals,
            depth=2 if subgoals else 1,
        )