from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThoughtStep:
    """A single step in a reasoning chain"""
    description: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """A structured chain of reasoning steps"""
    steps: List[ThoughtStep]
    overall_confidence: float
    domain: str = "general"
    created_at: datetime = field(default_factory=datetime.now)


class ThoughtProcessor:
    """Generates structured reasoning chains for goal processing.

    Unlike ThoughtAgent (which queries memory for past actions),
    ThoughtProcessor constructs fresh step-by-step reasoning chains
    that can then be validated and critiqued.
    """

    def __init__(self):
        self.chain_history: List[ReasoningChain] = []

    def generate(self, description: str, config: Dict[str, Any]) -> ReasoningChain:
        """Generate a reasoning chain for the given description.

        Args:
            description: The goal or problem to reason about.
            config: Options including 'domain', 'complexity_level', 'required_evidence'.

        Returns:
            A ReasoningChain with ordered ThoughtStep objects.
        """
        domain = config.get("domain", "general")
        complexity = max(1, int(config.get("complexity_level", 1)))
        required_evidence = max(1, int(config.get("required_evidence", 2)))

        steps = self._build_steps(description, complexity, required_evidence)
        overall_confidence = (
            sum(s.confidence for s in steps) / len(steps) if steps else 0.0
        )

        chain = ReasoningChain(
            steps=steps,
            overall_confidence=round(overall_confidence, 4),
            domain=domain,
        )
        self.chain_history.append(chain)
        logger.debug(
            "Generated reasoning chain: %d steps, confidence=%.2f",
            len(steps),
            overall_confidence,
        )
        return chain

    def generate_reasoning(
        self, description: str, context: Dict[str, Any]
    ) -> ReasoningChain:
        """Generate a reasoning chain using a context dict (alias for generate).

        Args:
            description: The goal or problem to reason about.
            context: Context dict; 'domain', 'complexity_level', 'required_evidence'
                     keys are forwarded to generate().

        Returns:
            A ReasoningChain.
        """
        config = {
            "domain": context.get("domain", "general"),
            "complexity_level": context.get("complexity_level", 1),
            "required_evidence": context.get("required_evidence", 2),
        }
        return self.generate(description, config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_steps(
        self, description: str, complexity: int, required_evidence: int
    ) -> List[ThoughtStep]:
        """Construct the ordered list of reasoning steps."""
        steps: List[ThoughtStep] = []

        # Step 1 – Problem analysis (always present)
        steps.append(
            ThoughtStep(
                description=f"Analyse the problem: {description}",
                confidence=0.90,
                evidence=["Problem statement provided"],
                metadata={"step_type": "analysis"},
            )
        )

        # Step 2 – Context and constraint evaluation
        steps.append(
            ThoughtStep(
                description=f"Evaluate constraints and context for: {description}",
                confidence=0.80,
                evidence=["Context and constraints reviewed"],
                metadata={"step_type": "evaluation"},
            )
        )

        # Middle steps – strategic reasoning layers scaled to complexity
        for layer in range(1, min(complexity, 5) + 1):
            layer_confidence = max(0.50, 0.80 - (layer - 1) * 0.08)
            evidence = [
                f"Evidence point {j + 1}" for j in range(required_evidence)
            ]
            steps.append(
                ThoughtStep(
                    description=(
                        f"Strategic reasoning layer {layer} for: {description}"
                    ),
                    confidence=layer_confidence,
                    evidence=evidence,
                    metadata={"step_type": "strategic", "layer": layer},
                )
            )

        # Final step – synthesis / conclusion
        steps.append(
            ThoughtStep(
                description=f"Synthesise findings and conclude for: {description}",
                confidence=0.75,
                evidence=["All prior steps considered"],
                metadata={"step_type": "conclusion"},
            )
        )

        return steps
