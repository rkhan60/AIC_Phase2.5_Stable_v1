from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

from .agents.goal_planner import GoalPlanner, Goal, GoalHierarchy
from .agents.intention_manager import IntentionManager, Intention
from .agents.thought_processor import ThoughtProcessor
from .agents.reasoning_validator import ReasoningValidator
from .agents.self_critic import SelfCritic
from .memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)

@dataclass
class AutonomousResult:
    """Result of autonomous execution cycle"""
    original_goal: Goal
    selected_intention: Intention
    reasoning_summary: str
    validation_status: str
    self_critic_summary: str
    memory_confirmation: bool
    metadata: Dict[str, Any]

@dataclass
class RetryHistory:
    """History of reasoning retries"""
    attempt_count: int
    feedback_history: List[str]
    intention_changes: List[str]
    final_scores: Dict[str, float]

@dataclass
class AutonomousCycleResult:
    original_goal: Optional[Goal]
    selected_intention: Optional[Intention]
    reasoning_summary: str
    validation_status: str
    self_critic_summary: str
    retry_history: Optional[RetryHistory]
    metadata: Dict[str, Any]

class AutonomousPipeline:
    """Integrates all components for autonomous operation"""
    
    def __init__(self, memory_dir: str = "memory"):
        """Initialize the autonomous pipeline"""
        self.goal_planner = GoalPlanner()
        self.intention_manager = IntentionManager()
        self.thought_processor = ThoughtProcessor()
        self.validator = ReasoningValidator()
        self.critic = SelfCritic()
        self.memory = MemorySystem(memory_dir)
        
    def run_cycle(self,
                 goal_input: str,
                 context: Optional[Dict[str, Any]] = None) -> AutonomousResult:
        """
        Run a complete autonomous cycle
        
        Args:
            goal_input: User's goal description
            context: Optional execution context
            
        Returns:
            AutonomousResult with execution details
        """
        try:
            logger.info(f"Starting autonomous cycle for goal: {goal_input}")
            cycle_start = datetime.now()
            
            # Step 1: Plan goal hierarchy
            hierarchy = self.goal_planner.plan_goal(goal_input, context or {})
            logger.info(f"Created goal hierarchy with {len(hierarchy.subgoals)} subgoals")
            
            # Step 2: Select intention
            intention = self.intention_manager.process_goals(hierarchy)
            logger.info(f"Selected intention: {intention.goal.description}")
            
            # Step 3: Generate reasoning
            reasoning_chain = self.thought_processor.generate(
                intention.goal.description,
                {
                    "domain": context.get("domain", "general"),
                    "complexity_level": intention.goal.complexity,
                    "required_evidence": max(2, intention.goal.complexity)
                }
            )
            
            # Step 4: Validate reasoning
            validation_result = self.validator.validate(reasoning_chain, context or {})
            logger.info(f"Validation status: {validation_result.status}")
            
            # Step 5: Self critique
            critique = self.critic.analyze(validation_result)
            logger.info(f"Critique score: {critique.overall_score:.2f}")
            
            # Step 6: Store in memory
            memory_success = self.memory.store_episode({
                "input": goal_input,
                "chain": reasoning_chain,
                "validation": validation_result.status,
                "flaws": validation_result.flaws,
                "critique": critique.summary,
                "metrics": validation_result.metrics,
                "intention": {
                    "goal": intention.goal.__dict__,
                    "status": intention.status,
                    "priority": intention.priority_score
                },
                "context": context,
                "timestamp": datetime.now()
            })
            
            # Update intention status based on validation
            success = validation_result.status == "valid"
            self.intention_manager.complete_intention(intention, success)
            
            # Prepare result
            result = AutonomousResult(
                original_goal=hierarchy.main_goal,
                selected_intention=intention,
                reasoning_summary=self._summarize_reasoning(reasoning_chain),
                validation_status=validation_result.status,
                self_critic_summary=critique.summary,
                memory_confirmation=memory_success,
                metadata={
                    "cycle_duration": (datetime.now() - cycle_start).total_seconds(),
                    "validation_metrics": validation_result.metrics,
                    "critique_score": critique.overall_score,
                    "pending_intentions": len(self.intention_manager.queue.pending_intentions)
                }
            )
            
            logger.info("Completed autonomous cycle")
            return result
            
        except Exception as e:
            logger.error(f"Error in autonomous cycle: {str(e)}")
            return AutonomousResult(
                original_goal=Goal(
                    description=goal_input,
                    priority=1.0,
                    complexity=1,
                    dependencies=[],
                    estimated_time=5.0,
                    goal_id="error"
                ),
                selected_intention=Intention(
                    goal=Goal(
                        description="Error recovery",
                        priority=1.0,
                        complexity=1,
                        dependencies=[],
                        estimated_time=5.0,
                        goal_id="error"
                    ),
                    status="failed",
                    context={"error": str(e)},
                    priority_score=1.0,
                    execution_log=["Error in autonomous cycle"]
                ),
                reasoning_summary="Error in processing",
                validation_status="invalid",
                self_critic_summary="System error occurred",
                memory_confirmation=False,
                metadata={"error": str(e)}
            )
            
    def _summarize_reasoning(self, chain: List[Any]) -> str:
        """Generate a summary of the reasoning chain"""
        if not chain:
            return "No reasoning generated"
            
        steps = len(chain)
        confidence = sum(step.confidence for step in chain) / steps
        
        return (
            f"Generated {steps} reasoning steps with "
            f"average confidence of {confidence:.2f}"
        )
        
    def display_results(self, result: AutonomousResult):
        """Display results in a formatted way"""
        print("\n🤖 [AIC Autonomous Cycle]")
        print(f"Original Goal: {result.original_goal.description}")
        
        print("\nSelected Intention:")
        print(f"Description: {result.selected_intention.goal.description}")
        print(f"Priority: {result.selected_intention.priority_score:.2f}")
        print(f"Status: {result.selected_intention.status}")
        
        print("\nReasoning:")
        print(f"Summary: {result.reasoning_summary}")
        print(f"Validation: {result.validation_status}")
        
        print("\nCritique:")
        print(result.self_critic_summary)
        
        print(f"\n→ Memory Storage: {'Successful' if result.memory_confirmation else 'Failed'}")
        
        if "cycle_duration" in result.metadata:
            print(f"\nCycle Duration: {result.metadata['cycle_duration']:.2f} seconds")

def run_autonomous_cycle(goal_input: str, context: Dict) -> AutonomousCycleResult:
    """Run a complete autonomous reasoning cycle with feedback loops"""
    try:
        # Initialize components
        goal_planner = GoalPlanner()
        intention_manager = IntentionManager()
        thought_processor = ThoughtProcessor()
        reasoning_validator = ReasoningValidator()
        self_critic = SelfCritic()
        
        # Initialize retry tracking
        retry_history = RetryHistory(
            attempt_count=0,
            feedback_history=[],
            intention_changes=[],
            final_scores={}
        )
        
        # 1. Goal Planning
        goal_hierarchy = goal_planner.decompose_goal(goal_input, context)
        current_intention = None
        best_result = None
        max_retries = 3
        
        while retry_history.attempt_count < max_retries:
            logger.info(f"Starting reasoning attempt {retry_history.attempt_count + 1}")
            
            # 2. Intention Selection (first time or after failures)
            if not current_intention:
                current_intention = intention_manager.select_intention(
                    [goal_hierarchy.main_goal] + goal_hierarchy.subgoals,
                    context
                )
                if current_intention:
                    retry_history.intention_changes.append(
                        f"Selected intention: {current_intention.goal.description}"
                    )
            
            # 3. Reasoning Generation
            reasoning_context = {
                **context,
                "previous_feedback": retry_history.feedback_history[-1] if retry_history.feedback_history else None
            }
            
            reasoning_chain = thought_processor.generate_reasoning(
                current_intention.goal.description,
                reasoning_context
            )
            
            # 4. Validation
            validation_result = reasoning_validator.validate_chain(reasoning_chain)
            
            # 5. Self Critique
            critique_result = self_critic.analyze_reasoning(reasoning_chain, validation_result)
            
            # Track metrics
            current_scores = {
                "validation_score": validation_result.confidence_score,
                "critique_score": critique_result.overall_score
            }
            retry_history.final_scores.update(current_scores)
            
            # Check if reasoning is acceptable
            is_valid = validation_result.is_valid
            critique_acceptable = critique_result.overall_score >= 0.7
            
            if is_valid and critique_acceptable:
                best_result = (reasoning_chain, validation_result, critique_result)
                break
                
            # Prepare for retry
            retry_history.attempt_count += 1
            
            # Collect feedback for next attempt
            feedback = []
            if not is_valid:
                feedback.extend(validation_result.issues)
            if not critique_acceptable:
                feedback.extend([
                    f"{point.aspect}: {point.improvement}"
                    for point in critique_result.points
                ])
            
            retry_history.feedback_history.append("\n".join(feedback))
            
            # If max retries reached with current intention, try a new one
            if retry_history.attempt_count == 2:
                logger.info("Switching to alternate intention after repeated failures")
                current_intention = None  # Will trigger new intention selection
                retry_history.attempt_count = 0  # Reset for new intention attempts
        
        # Use best result or last attempt
        if best_result:
            reasoning_chain, validation_result, critique_result = best_result
        
        # Prepare final result
        return AutonomousCycleResult(
            original_goal=goal_hierarchy.main_goal,
            selected_intention=current_intention,
            reasoning_summary="\n".join([step.description for step in reasoning_chain.steps]),
            validation_status="valid" if validation_result.is_valid else "invalid",
            self_critic_summary="\n".join([
                f"{point.aspect}: {point.observation} - {point.improvement}"
                for point in critique_result.points
            ]),
            retry_history=retry_history,
            metadata={
                "overall_confidence": reasoning_chain.overall_confidence,
                "validation_score": validation_result.confidence_score,
                "critique_score": critique_result.overall_score,
                "total_attempts": retry_history.attempt_count,
                "intention_changes": len(retry_history.intention_changes)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in autonomous cycle: {str(e)}")
        return AutonomousCycleResult(
            original_goal=None,
            selected_intention=None,
            reasoning_summary=f"Error: {str(e)}",
            validation_status="error",
            self_critic_summary="Could not complete analysis due to error",
            retry_history=None,
            metadata={"error": str(e)}
        )

# Example usage
if __name__ == "__main__":
    # Test goal
    goal = "Develop a machine learning model for customer churn prediction"
    
    # Optional context
    context = {
        "domain": "ML Engineering",
        "time_constraint": 120,
        "required_accuracy": 0.85
    }
    
    # Run autonomous cycle
    result = run_autonomous_cycle(goal, context)
    
    # Results are also returned as an object
    print("\nRaw Result Data:")
    print(f"Goal: {result.original_goal.description}")
    print(f"Intention Status: {result.selected_intention.status}")
    print(f"Memory Stored: {result.memory_confirmation}") 