from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging
from ..memory.memory_query import MemoryQueryEngine

logger = logging.getLogger(__name__)


@dataclass
class CritiquePoint:
    """A single observation/suggestion produced by the self-critic."""
    aspect: str
    observation: str
    improvement: str


@dataclass
class CritiqueResult:
    """High-level result returned by analyze() and analyze_reasoning()."""
    overall_score: float
    summary: str
    points: List[CritiquePoint] = field(default_factory=list)

class CritiqueType(Enum):
    CLARITY = "clarity"        # Clarity of reasoning
    CONSISTENCY = "consistency"  # Internal consistency
    COMPLETENESS = "completeness"  # Completeness of analysis
    EFFICIENCY = "efficiency"    # Resource efficiency
    EFFECTIVENESS = "effectiveness"  # Goal achievement

@dataclass
class CritiqueMetrics:
    """Metrics for critique evaluation"""
    clarity_score: float = 0.0
    consistency_score: float = 0.0
    completeness_score: float = 0.0
    efficiency_score: float = 0.0
    effectiveness_score: float = 0.0
    
    def get_overall_score(self) -> float:
        """Calculate overall critique score"""
        weights = {
            'clarity': 0.2,
            'consistency': 0.2,
            'completeness': 0.2,
            'efficiency': 0.2,
            'effectiveness': 0.2
        }
        scores = {
            'clarity': self.clarity_score,
            'consistency': self.consistency_score,
            'completeness': self.completeness_score,
            'efficiency': self.efficiency_score,
            'effectiveness': self.effectiveness_score
        }
        return sum(score * weights[metric] for metric, score in scores.items())

@dataclass
class Critique:
    """Represents a critique of reasoning or action"""
    id: str
    type: CritiqueType
    target_id: str  # ID of critiqued item
    metrics: CritiqueMetrics
    findings: List[str]
    suggestions: List[str]
    priority: float
    created_at: datetime = field(default_factory=datetime.now)

class SelfCritic:
    """Agent for evaluating reasoning and suggesting improvements"""
    def __init__(self, memory_query: MemoryQueryEngine):
        self.memory_query = memory_query
        self.critiques: Dict[str, Critique] = {}
        self.critique_history: List[Dict[str, Any]] = []
        
    def evaluate_reasoning(self,
                         reasoning_trace: List[Dict[str, Any]],
                         context: Dict[str, Any]) -> Critique:
        """Evaluate reasoning trace and generate critique"""
        # Generate critique ID
        critique_id = self._generate_critique_id(reasoning_trace)
        
        # Evaluate metrics
        metrics = self._evaluate_critique_metrics(reasoning_trace, context)
        
        # Analyze reasoning
        findings = self._analyze_reasoning(reasoning_trace, metrics)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(findings, context)
        
        # Calculate priority
        priority = self._calculate_priority(metrics, findings)
        
        # Create critique
        critique = Critique(
            id=critique_id,
            type=CritiqueType.CLARITY,  # Default type
            target_id=str(reasoning_trace[0].get('id', 'unknown')),
            metrics=metrics,
            findings=findings,
            suggestions=suggestions,
            priority=priority
        )
        
        # Document critique
        self._log_critique(critique, reasoning_trace)
        
        # Store critique
        self.critiques[critique_id] = critique
        
        return critique
        
    def evaluate_action(self,
                       action_trace: List[Dict[str, Any]],
                       expected_outcomes: List[str],
                       context: Dict[str, Any]) -> Critique:
        """Evaluate action execution and outcomes"""
        # Generate critique ID
        critique_id = self._generate_critique_id(action_trace)
        
        # Evaluate metrics
        metrics = self._evaluate_action_metrics(action_trace, expected_outcomes)
        
        # Analyze execution
        findings = self._analyze_execution(action_trace, expected_outcomes)
        
        # Generate suggestions
        suggestions = self._generate_action_suggestions(findings, context)
        
        # Calculate priority
        priority = self._calculate_priority(metrics, findings)
        
        # Create critique
        critique = Critique(
            id=critique_id,
            type=CritiqueType.EFFECTIVENESS,
            target_id=str(action_trace[0].get('id', 'unknown')),
            metrics=metrics,
            findings=findings,
            suggestions=suggestions,
            priority=priority
        )
        
        # Document critique
        self._log_critique(critique, action_trace)
        
        # Store critique
        self.critiques[critique_id] = critique
        
        return critique
        
    def _generate_critique_id(self, trace: List[Dict[str, Any]]) -> str:
        """Generate unique critique ID"""
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_hash = hashlib.md5(str(trace).encode()).hexdigest()[:8]
        return f"critique_{timestamp}_{trace_hash}"
        
    def _evaluate_critique_metrics(self,
                                 reasoning_trace: List[Dict[str, Any]],
                                 context: Dict[str, Any]) -> CritiqueMetrics:
        """Evaluate critique metrics for reasoning"""
        metrics = CritiqueMetrics()
        
        # Evaluate clarity
        metrics.clarity_score = self._evaluate_clarity(reasoning_trace)
        
        # Evaluate consistency
        metrics.consistency_score = self._evaluate_consistency(reasoning_trace)
        
        # Evaluate completeness
        metrics.completeness_score = self._evaluate_completeness(reasoning_trace)
        
        # Evaluate efficiency
        metrics.efficiency_score = self._evaluate_efficiency(reasoning_trace)
        
        # Evaluate effectiveness
        metrics.effectiveness_score = self._evaluate_effectiveness(reasoning_trace)
        
        return metrics
        
    def _evaluate_action_metrics(self,
                               action_trace: List[Dict[str, Any]],
                               expected_outcomes: List[str]) -> CritiqueMetrics:
        """Evaluate critique metrics for action"""
        metrics = CritiqueMetrics()
        
        # Evaluate clarity
        metrics.clarity_score = self._evaluate_action_clarity(action_trace)
        
        # Evaluate consistency
        metrics.consistency_score = self._evaluate_action_consistency(action_trace)
        
        # Evaluate completeness
        metrics.completeness_score = self._evaluate_action_completeness(
            action_trace, expected_outcomes)
        
        # Evaluate efficiency
        metrics.efficiency_score = self._evaluate_action_efficiency(action_trace)
        
        # Evaluate effectiveness
        metrics.effectiveness_score = self._evaluate_action_effectiveness(
            action_trace, expected_outcomes)
        
        return metrics
        
    def _evaluate_clarity(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate reasoning clarity"""
        if not trace:
            return 0.0
            
        factors = [
            bool(step.get('rationale')) for step in trace
        ]
        return sum(1 for f in factors if f) / len(factors)
        
    def _evaluate_consistency(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate reasoning consistency"""
        if not trace:
            return 0.0
            
        # Check for contradictions in reasoning
        return 0.8  # Placeholder implementation
        
    def _evaluate_completeness(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate reasoning completeness"""
        if not trace:
            return 0.0
            
        required_fields = ['timestamp', 'action', 'rationale']
        completeness = sum(
            all(field in step for field in required_fields)
            for step in trace
        )
        return completeness / len(trace)
        
    def _evaluate_efficiency(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate reasoning efficiency"""
        if not trace:
            return 0.0
            
        # Analyze reasoning steps for redundancy
        return 0.7  # Placeholder implementation
        
    def _evaluate_effectiveness(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate reasoning effectiveness"""
        if not trace:
            return 0.0
            
        # Analyze outcome achievement
        return 0.75  # Placeholder implementation
        
    def _evaluate_action_clarity(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate action clarity"""
        if not trace:
            return 0.0
            
        factors = [
            bool(step.get('action')),
            bool(step.get('parameters')),
            bool(step.get('rationale'))
        ]
        return sum(1 for f in factors if f) / len(factors)
        
    def _evaluate_action_consistency(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate action consistency"""
        if not trace:
            return 0.0
            
        # Check for action sequence consistency
        return 0.85  # Placeholder implementation
        
    def _evaluate_action_completeness(self,
                                    trace: List[Dict[str, Any]],
                                    expected_outcomes: List[str]) -> float:
        """Evaluate action completeness"""
        if not trace or not expected_outcomes:
            return 0.0
            
        # Check if all expected outcomes were addressed
        addressed_outcomes = sum(
            any(outcome in str(step) for step in trace)
            for outcome in expected_outcomes
        )
        return addressed_outcomes / len(expected_outcomes)
        
    def _evaluate_action_efficiency(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate action efficiency"""
        if not trace:
            return 0.0
            
        # Analyze action sequence for redundancy
        return 0.9  # Placeholder implementation
        
    def _evaluate_action_effectiveness(self,
                                     trace: List[Dict[str, Any]],
                                     expected_outcomes: List[str]) -> float:
        """Evaluate action effectiveness"""
        if not trace or not expected_outcomes:
            return 0.0
            
        # Compare actual vs expected outcomes
        return 0.8  # Placeholder implementation
        
    def _analyze_reasoning(self,
                         trace: List[Dict[str, Any]],
                         metrics: CritiqueMetrics) -> List[str]:
        """Analyze reasoning trace and generate findings"""
        findings = []
        
        # Check clarity
        if metrics.clarity_score < 0.7:
            findings.append("Reasoning lacks clear explanation in some steps")
            
        # Check consistency
        if metrics.consistency_score < 0.7:
            findings.append("Some reasoning steps show inconsistencies")
            
        # Check completeness
        if metrics.completeness_score < 0.7:
            findings.append("Reasoning missing key information or steps")
            
        return findings
        
    def _analyze_execution(self,
                         trace: List[Dict[str, Any]],
                         expected_outcomes: List[str]) -> List[str]:
        """Analyze action execution"""
        findings = []
        
        # Check execution completeness
        if not all(any(outcome in str(step) for step in trace)
                  for outcome in expected_outcomes):
            findings.append("Not all expected outcomes were achieved")
            
        # Check execution efficiency
        if len(trace) > len(expected_outcomes) * 2:
            findings.append("Execution contains possible redundant steps")
            
        return findings
        
    def _generate_suggestions(self,
                            findings: List[str],
                            context: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        for finding in findings:
            if "lacks clear explanation" in finding:
                suggestions.append(
                    "Add explicit rationale for each reasoning step")
            elif "inconsistencies" in finding:
                suggestions.append(
                    "Review and align reasoning chain for consistency")
            elif "missing key information" in finding:
                suggestions.append(
                    "Ensure all required fields are populated")
                
        return suggestions
        
    def _generate_action_suggestions(self,
                                   findings: List[str],
                                   context: Dict[str, Any]) -> List[str]:
        """Generate action improvement suggestions"""
        suggestions = []
        
        for finding in findings:
            if "outcomes were achieved" in finding:
                suggestions.append(
                    "Review and address missing expected outcomes")
            elif "redundant steps" in finding:
                suggestions.append(
                    "Optimize action sequence for efficiency")
                
        return suggestions
        
    def _calculate_priority(self,
                          metrics: CritiqueMetrics,
                          findings: List[str]) -> float:
        """Calculate critique priority"""
        # Base priority on overall metrics score
        base_priority = 1.0 - metrics.get_overall_score()
        
        # Adjust based on number of findings
        finding_factor = len(findings) / 10  # Normalize to 0-1 range
        
        return min(1.0, base_priority + finding_factor * 0.3)
        
    def _log_critique(self,
                     critique: Critique,
                     trace: List[Dict[str, Any]]):
        """Log critique details"""
        self.critique_history.append({
            'timestamp': datetime.now(),
            'critique_id': critique.id,
            'target_id': critique.target_id,
            'metrics': critique.metrics.__dict__,
            'findings': critique.findings,
            'suggestions': critique.suggestions,
            'priority': critique.priority,
            'trace_length': len(trace)
        })
        
    def get_critique_history(self,
                           target_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get critique history for target"""
        if not target_id:
            return self.critique_history

        return [
            critique for critique in self.critique_history
            if critique['target_id'] == target_id
        ]

    # ------------------------------------------------------------------
    # Pipeline-facing API (used by AutonomousPipeline)
    # ------------------------------------------------------------------

    def analyze(self, validation_result: Any) -> CritiqueResult:
        """Analyse a ValidationResult and return a CritiqueResult.

        Args:
            validation_result: A ValidationResult from ReasoningValidator.

        Returns:
            CritiqueResult with score, summary and improvement points.
        """
        flaws = getattr(validation_result, 'flaws', [])
        confidence = getattr(validation_result, 'confidence_score', 0.75)
        is_valid = getattr(validation_result, 'is_valid', False)

        points: List[CritiquePoint] = []
        for flaw in flaws:
            points.append(CritiquePoint(
                aspect="Validation",
                observation=flaw,
                improvement="Address the identified flaw before proceeding"
            ))

        overall_score = min(1.0, confidence + (0.1 if is_valid else 0.0))
        summary = (
            f"Validation {'passed' if is_valid else 'failed'} with confidence "
            f"{confidence:.2f}. Found {len(flaws)} flaw(s)."
        )

        return CritiqueResult(
            overall_score=round(overall_score, 4),
            summary=summary,
            points=points,
        )

    def analyze_reasoning(
        self, reasoning_chain: Any, validation_result: Any
    ) -> CritiqueResult:
        """Analyse a ReasoningChain together with its ValidationResult.

        Args:
            reasoning_chain: A ReasoningChain from ThoughtProcessor.
            validation_result: The corresponding ValidationResult.

        Returns:
            CritiqueResult combining reasoning quality and validation findings.
        """
        steps = getattr(reasoning_chain, 'steps', [])
        overall_confidence = getattr(reasoning_chain, 'overall_confidence', 0.75)
        flaws = getattr(validation_result, 'flaws', [])

        # Build a trace compatible with evaluate_reasoning
        trace = [
            {
                'rationale': s.description,
                'context': s.metadata,
                'analysis': s.evidence,
                'confidence': s.confidence,
            }
            for s in steps
        ]

        points: List[CritiquePoint] = []

        if trace:
            try:
                critique = self.evaluate_reasoning(
                    trace,
                    {'validation_status': getattr(validation_result, 'status', 'unknown')}
                )
                for i, finding in enumerate(critique.findings):
                    suggestion = (
                        critique.suggestions[i]
                        if i < len(critique.suggestions)
                        else "Review and improve"
                    )
                    points.append(CritiquePoint(
                        aspect="Reasoning Quality",
                        observation=finding,
                        improvement=suggestion,
                    ))
            except Exception as exc:
                logger.warning("evaluate_reasoning raised: %s", exc)

        for flaw in flaws:
            points.append(CritiquePoint(
                aspect="Validation Flaw",
                observation=flaw,
                improvement="Strengthen this aspect of the reasoning chain",
            ))

        summary = (
            f"Reasoning chain: {len(steps)} step(s), "
            f"confidence {overall_confidence:.2f}. "
            f"{len(points)} improvement area(s) identified."
        )

        return CritiqueResult(
            overall_score=round(overall_confidence, 4),
            summary=summary,
            points=points,
        )