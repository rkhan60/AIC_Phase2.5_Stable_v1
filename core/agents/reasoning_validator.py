from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging
from .thought_processor import ReasoningChain

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a reasoning chain validation pass."""
    is_valid: bool
    status: str          # "valid" | "invalid"
    confidence_score: float
    flaws: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class ReasoningValidator:
    """Validates reasoning chains for logical consistency and completeness.

    Checks:
    - Minimum number of reasoning steps
    - Per-step confidence thresholds
    - Evidence coverage across steps
    """

    MIN_STEPS = 2
    MIN_STEP_CONFIDENCE = 0.50

    def validate(
        self, chain: ReasoningChain, context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate a reasoning chain (context-aware entry point).

        Args:
            chain: The ReasoningChain to evaluate.
            context: Execution context (currently used for future extensions).

        Returns:
            A ValidationResult describing validity, flaws, and metrics.
        """
        return self._run_validation(chain)

    def validate_chain(self, chain: ReasoningChain) -> ValidationResult:
        """Validate a reasoning chain without context."""
        return self._run_validation(chain)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_validation(self, chain: ReasoningChain) -> ValidationResult:
        flaws: List[str] = []
        issues: List[str] = []

        # Check 1 – minimum steps
        if not chain.steps or len(chain.steps) < self.MIN_STEPS:
            flaws.append(
                f"Insufficient reasoning steps "
                f"(found {len(chain.steps)}, need {self.MIN_STEPS})"
            )
            issues.append(
                "Add more reasoning steps to adequately support the conclusion"
            )

        # Check 2 – per-step confidence
        low_confidence = [
            s for s in chain.steps if s.confidence < self.MIN_STEP_CONFIDENCE
        ]
        if low_confidence:
            flaws.append(
                f"{len(low_confidence)} step(s) have confidence below "
                f"{self.MIN_STEP_CONFIDENCE:.0%}"
            )
            issues.append(
                "Review and strengthen low-confidence reasoning steps"
            )

        # Check 3 – evidence coverage
        no_evidence = [s for s in chain.steps if not s.evidence]
        if no_evidence:
            flaws.append(
                f"{len(no_evidence)} step(s) lack supporting evidence"
            )
            issues.append("Provide evidence for every reasoning step")

        total_steps = len(chain.steps)
        evidence_coverage = (
            1.0 - len(no_evidence) / total_steps if total_steps > 0 else 0.0
        )
        confidence_score = chain.overall_confidence if chain.steps else 0.0

        metrics: Dict[str, float] = {
            "step_count": float(total_steps),
            "avg_confidence": confidence_score,
            "evidence_coverage": round(evidence_coverage, 4),
            "flaw_count": float(len(flaws)),
        }

        is_valid = len(flaws) == 0
        logger.debug(
            "Validation complete: valid=%s, flaws=%d, confidence=%.2f",
            is_valid,
            len(flaws),
            confidence_score,
        )
        return ValidationResult(
            is_valid=is_valid,
            status="valid" if is_valid else "invalid",
            confidence_score=round(confidence_score, 4),
            flaws=flaws,
            issues=issues,
            metrics=metrics,
        )
