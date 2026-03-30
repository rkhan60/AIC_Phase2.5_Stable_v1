from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from .agents.goal_planner import GoalPlanner, Goal, GoalHierarchy, GoalType, GoalMetrics
from .agents.intention_manager import (
    IntentionManager,
    Intention,
    IntentionStatus,
    IntentionMetrics,
)
from .agents.thought_processor import ThoughtProcessor
from .agents.reasoning_validator import ReasoningValidator
from .agents.self_critic import SelfCritic
from .memory.memory_system import MemorySystem
from .memory.memory_query import MemoryQueryEngine

logger = logging.getLogger(__name__)


@dataclass
class AutonomousResult:
    """Result of a single autonomous execution cycle."""
    original_goal: Goal
    selected_intention: Intention
    reasoning_summary: str
    validation_status: str
    self_critic_summary: str
    memory_confirmation: bool
    metadata: Dict[str, Any]


@dataclass
class RetryHistory:
    """Tracks reasoning retries within a cycle."""
    attempt_count: int
    feedback_history: List[str]
    intention_changes: List[str]
    final_scores: Dict[str, float]


@dataclass
class AutonomousCycleResult:
    """Result of the standalone run_autonomous_cycle() function."""
    original_goal: Optional[Goal]
    selected_intention: Optional[Intention]
    reasoning_summary: str
    validation_status: str
    self_critic_summary: str
    retry_history: Optional[RetryHistory]
    metadata: Dict[str, Any]


class AutonomousPipeline:
    """Integrates all components for autonomous goal processing."""

    def __init__(self, memory_dir: str = "memory"):
        """Initialise the autonomous pipeline.

        All agents share a single MemoryQueryEngine so they operate on the
        same semantic index.

        Args:
            memory_dir: Directory path for persistent memory storage.
        """
        memory_query = MemoryQueryEngine()
        self.goal_planner = GoalPlanner(memory_query)
        self.intention_manager = IntentionManager(memory_query)
        self.thought_processor = ThoughtProcessor()
        self.validator = ReasoningValidator()
        self.critic = SelfCritic(memory_query)
        self.memory = MemorySystem(storage_path=Path(memory_dir))

    def run_cycle(
        self,
        goal_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AutonomousResult:
        """Run a complete autonomous reasoning cycle.

        Steps:
          1. Plan goal hierarchy
          2. Select primary intention
          3. Generate reasoning chain
          4. Validate reasoning
          5. Self-critique
          6. Store episode in memory

        Args:
            goal_input: Free-text description of the user's goal.
            context: Optional execution context (domain, constraints, etc.).

        Returns:
            AutonomousResult with all cycle outputs.
        """
        context = context or {}
        cycle_start = datetime.now()

        try:
            logger.info("Starting autonomous cycle for goal: %s", goal_input)

            # Step 1 – Goal planning
            hierarchy = self.goal_planner.plan_goal(goal_input, context)
            logger.info(
                "Created goal hierarchy with %d subgoal(s)",
                len(hierarchy.subgoals),
            )

            # Step 2 – Intention selection
            intention = self.intention_manager.process_goals(hierarchy, context)
            goal = self.intention_manager.active_goals.get(intention.goal_id)
            goal_description = goal.description if goal else goal_input
            logger.info("Selected intention for: %s", goal_description)

            # Step 3 – Reasoning generation
            reasoning_chain = self.thought_processor.generate(
                goal_description,
                {
                    "domain": context.get("domain", "general"),
                    "complexity_level": context.get("complexity_level", 1),
                    "required_evidence": max(
                        2, int(context.get("complexity_level", 1))
                    ),
                },
            )

            # Step 4 – Validation
            validation_result = self.validator.validate(reasoning_chain, context)
            logger.info("Validation status: %s", validation_result.status)

            # Step 5 – Self-critique
            critique = self.critic.analyze_reasoning(reasoning_chain, validation_result)
            logger.info("Critique score: %.2f", critique.overall_score)

            # Step 6 – Store episode
            memory_success = self.memory.store_episode(
                {
                    "input": goal_input,
                    "chain": [s.__dict__ for s in reasoning_chain.steps],
                    "validation": validation_result.status,
                    "flaws": validation_result.flaws,
                    "critique": critique.summary,
                    "metrics": validation_result.metrics,
                    "intention": {
                        "goal_id": intention.goal_id,
                        "status": intention.status.value,
                        "priority": intention.metrics.priority,
                    },
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Update intention status
            if validation_result.status == "valid":
                self.intention_manager.complete_intention(intention.id)

            active_count = sum(
                1
                for i in self.intention_manager.intentions.values()
                if i.status == IntentionStatus.ACTIVE
            )

            return AutonomousResult(
                original_goal=hierarchy.main_goal,
                selected_intention=intention,
                reasoning_summary=self._summarize_reasoning(reasoning_chain.steps),
                validation_status=validation_result.status,
                self_critic_summary=critique.summary,
                memory_confirmation=memory_success,
                metadata={
                    "cycle_duration": (
                        datetime.now() - cycle_start
                    ).total_seconds(),
                    "validation_metrics": validation_result.metrics,
                    "critique_score": critique.overall_score,
                    "active_intentions": active_count,
                },
            )

        except Exception as exc:
            logger.error("Error in autonomous cycle: %s", exc, exc_info=True)
            placeholder_goal = Goal(
                id="error",
                type=GoalType.IMMEDIATE,
                description=goal_input,
                success_criteria=[],
            )
            placeholder_intention = Intention(
                id="error",
                goal_id="error",
                status=IntentionStatus.FAILED,
                metrics=IntentionMetrics(),
            )
            return AutonomousResult(
                original_goal=placeholder_goal,
                selected_intention=placeholder_intention,
                reasoning_summary="Error in processing",
                validation_status="invalid",
                self_critic_summary="System error occurred",
                memory_confirmation=False,
                metadata={"error": str(exc)},
            )

    def _summarize_reasoning(self, steps: list) -> str:
        """Produce a one-line summary of a reasoning chain."""
        if not steps:
            return "No reasoning generated"
        avg_confidence = sum(s.confidence for s in steps) / len(steps)
        return (
            f"Generated {len(steps)} reasoning step(s) with "
            f"average confidence of {avg_confidence:.2f}"
        )

    def display_results(self, result: AutonomousResult) -> None:
        """Print a formatted summary of an AutonomousResult."""
        print("\n[AIC Autonomous Cycle]")
        print(f"Original Goal : {result.original_goal.description}")

        intention = result.selected_intention
        goal = self.intention_manager.active_goals.get(intention.goal_id)
        print("\nSelected Intention:")
        print(f"  Description : {goal.description if goal else 'N/A'}")
        print(f"  Priority    : {intention.metrics.priority:.2f}")
        print(f"  Status      : {intention.status.value}")

        print("\nReasoning:")
        print(f"  Summary    : {result.reasoning_summary}")
        print(f"  Validation : {result.validation_status}")

        print("\nCritique:")
        print(f"  {result.self_critic_summary}")

        stored = "Successful" if result.memory_confirmation else "Failed"
        print(f"\n-> Memory Storage: {stored}")

        duration = result.metadata.get("cycle_duration")
        if duration is not None:
            print(f"   Cycle Duration: {duration:.2f}s")


def run_autonomous_cycle(
    goal_input: str, context: Dict[str, Any]
) -> AutonomousCycleResult:
    """Run a complete autonomous reasoning cycle with retry logic.

    This standalone function mirrors AutonomousPipeline.run_cycle() but adds
    a feedback-loop that retries up to 3 times before switching intentions.

    Args:
        goal_input: Free-text description of the goal.
        context: Execution context (domain, constraints, etc.).

    Returns:
        AutonomousCycleResult with the best result found.
    """
    try:
        # Initialise all components with shared memory query engine
        memory_query = MemoryQueryEngine()
        goal_planner = GoalPlanner(memory_query)
        intention_manager = IntentionManager(memory_query)
        thought_processor = ThoughtProcessor()
        reasoning_validator = ReasoningValidator()
        self_critic = SelfCritic(memory_query)

        retry_history = RetryHistory(
            attempt_count=0,
            feedback_history=[],
            intention_changes=[],
            final_scores={},
        )

        # 1 – Goal planning
        goal_hierarchy = goal_planner.plan_goal(goal_input, context)

        current_intention: Optional[Intention] = None
        best_result = None
        max_retries = 3

        while retry_history.attempt_count < max_retries:
            logger.info(
                "Reasoning attempt %d", retry_history.attempt_count + 1
            )

            # 2 – Intention selection (first time or after failures)
            if current_intention is None:
                candidates = [goal_hierarchy.main_goal] + goal_hierarchy.subgoals
                current_intention = intention_manager.select_intention(
                    candidates, context
                )
                if current_intention:
                    selected_goal = intention_manager.active_goals.get(
                        current_intention.goal_id
                    )
                    retry_history.intention_changes.append(
                        f"Selected intention: "
                        f"{selected_goal.description if selected_goal else 'unknown'}"
                    )

            if current_intention is None:
                break

            selected_goal = intention_manager.active_goals.get(
                current_intention.goal_id
            )
            goal_description = (
                selected_goal.description if selected_goal else goal_input
            )

            # 3 – Reasoning generation
            reasoning_context = {
                **context,
                "previous_feedback": (
                    retry_history.feedback_history[-1]
                    if retry_history.feedback_history
                    else None
                ),
            }
            reasoning_chain = thought_processor.generate_reasoning(
                goal_description, reasoning_context
            )

            # 4 – Validation
            validation_result = reasoning_validator.validate_chain(reasoning_chain)

            # 5 – Self-critique
            critique_result = self_critic.analyze_reasoning(
                reasoning_chain, validation_result
            )

            retry_history.final_scores.update(
                {
                    "validation_score": validation_result.confidence_score,
                    "critique_score": critique_result.overall_score,
                }
            )

            is_valid = validation_result.is_valid
            critique_ok = critique_result.overall_score >= 0.7

            if is_valid and critique_ok:
                best_result = (reasoning_chain, validation_result, critique_result)
                break

            retry_history.attempt_count += 1

            # Collect feedback for next attempt
            feedback = list(validation_result.issues)
            feedback += [
                f"{p.aspect}: {p.improvement}" for p in critique_result.points
            ]
            retry_history.feedback_history.append("\n".join(feedback))

            # After 2 failed attempts with the same intention, try another
            if retry_history.attempt_count == 2:
                logger.info("Switching to alternate intention after repeated failures")
                current_intention = None
                retry_history.attempt_count = 0

        if best_result is None and current_intention is not None:
            # Use the last attempt if nothing passed
            best_result = (reasoning_chain, validation_result, critique_result)

        if best_result is None:
            raise RuntimeError("No reasoning result produced")

        reasoning_chain, validation_result, critique_result = best_result

        reasoning_summary = " | ".join(
            s.description for s in reasoning_chain.steps
        )
        critic_summary = "\n".join(
            f"{p.aspect}: {p.observation} -> {p.improvement}"
            for p in critique_result.points
        )

        return AutonomousCycleResult(
            original_goal=goal_hierarchy.main_goal,
            selected_intention=current_intention,
            reasoning_summary=reasoning_summary,
            validation_status="valid" if validation_result.is_valid else "invalid",
            self_critic_summary=critic_summary or critique_result.summary,
            retry_history=retry_history,
            metadata={
                "overall_confidence": reasoning_chain.overall_confidence,
                "validation_score": validation_result.confidence_score,
                "critique_score": critique_result.overall_score,
                "total_attempts": retry_history.attempt_count,
                "intention_changes": len(retry_history.intention_changes),
            },
        )

    except Exception as exc:
        logger.error("Error in autonomous cycle: %s", exc, exc_info=True)
        return AutonomousCycleResult(
            original_goal=None,
            selected_intention=None,
            reasoning_summary=f"Error: {exc}",
            validation_status="error",
            self_critic_summary="Could not complete analysis due to error",
            retry_history=None,
            metadata={"error": str(exc)},
        )


if __name__ == "__main__":
    goal = "Develop a machine learning model for customer churn prediction"
    context = {
        "domain": "ML Engineering",
        "time_constraint": 120,
        "required_accuracy": 0.85,
    }

    result = run_autonomous_cycle(goal, context)

    print("\nAIC Autonomous Cycle Result")
    print(f"Goal       : {result.original_goal.description if result.original_goal else 'N/A'}")
    print(f"Validation : {result.validation_status}")
    print(f"Summary    : {result.reasoning_summary[:200]}")
