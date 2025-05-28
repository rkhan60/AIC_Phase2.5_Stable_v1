from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging
from ..memory.memory_query import MemoryQueryEngine

logger = logging.getLogger(__name__)

class ScoreCategory(Enum):
    CLARITY = "clarity"          # Clear and understandable
    RELEVANCE = "relevance"      # Relevant to goals/context
    VALUE = "value"              # Value/utility of decision
    COHERENCE = "coherence"      # Internal consistency
    ADAPTABILITY = "adaptability"  # Ability to adapt

@dataclass
class ReasoningMetrics:
    """Metrics for reasoning evaluation"""
    clarity: float = 0.0       # How clear and well-structured
    relevance: float = 0.0     # How relevant to context
    value: float = 0.0         # Expected value/utility
    coherence: float = 0.0     # Internal consistency
    adaptability: float = 0.0  # Adaptability to change
    
    def get_overall_score(self) -> float:
        """Calculate overall reasoning score"""
        weights = {
            'clarity': 0.2,
            'relevance': 0.25,
            'value': 0.3,
            'coherence': 0.15,
            'adaptability': 0.1
        }
        scores = {
            'clarity': self.clarity,
            'relevance': self.relevance,
            'value': self.value,
            'coherence': self.coherence,
            'adaptability': self.adaptability
        }
        return sum(score * weights[metric] for metric, score in scores.items())

@dataclass
class ReasoningScore:
    """Represents a reasoning evaluation score"""
    id: str
    target_id: str
    metrics: ReasoningMetrics
    details: Dict[str, Any]
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)

class ReasoningScorer:
    """Agent for evaluating reasoning quality"""
    def __init__(self, memory_query: MemoryQueryEngine):
        self.memory_query = memory_query
        self.scores: Dict[str, ReasoningScore] = {}
        self.score_history: List[Dict[str, Any]] = []
        
    def evaluate_reasoning(self,
                         reasoning_trace: List[Dict[str, Any]],
                         context: Dict[str, Any]) -> ReasoningScore:
        """Evaluate reasoning quality"""
        # Generate score ID
        score_id = self._generate_score_id(reasoning_trace)
        
        # Evaluate metrics
        metrics = self._evaluate_metrics(reasoning_trace, context)
        
        # Analyze details
        details = self._analyze_details(reasoning_trace, metrics)
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics, details)
        
        # Create score
        score = ReasoningScore(
            id=score_id,
            target_id=str(reasoning_trace[0].get('id', 'unknown')),
            metrics=metrics,
            details=details,
            confidence=confidence
        )
        
        # Document score
        self._log_score(score, reasoning_trace)
        
        # Store score
        self.scores[score_id] = score
        
        return score
        
    def evaluate_decision(self,
                         decision: Dict[str, Any],
                         context: Dict[str, Any],
                         expected_outcomes: List[str]) -> ReasoningScore:
        """Evaluate decision quality"""
        # Generate score ID
        score_id = self._generate_score_id([decision])
        
        # Create pseudo reasoning trace
        decision_trace = [{
            'id': decision.get('id', 'unknown'),
            'type': 'decision',
            'content': decision,
            'expected_outcomes': expected_outcomes
        }]
        
        # Evaluate metrics
        metrics = self._evaluate_decision_metrics(decision, context, expected_outcomes)
        
        # Analyze details
        details = self._analyze_decision_details(decision, expected_outcomes)
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics, details)
        
        # Create score
        score = ReasoningScore(
            id=score_id,
            target_id=str(decision.get('id', 'unknown')),
            metrics=metrics,
            details=details,
            confidence=confidence
        )
        
        # Document score
        self._log_score(score, decision_trace)
        
        # Store score
        self.scores[score_id] = score
        
        return score
        
    def _generate_score_id(self, trace: List[Dict[str, Any]]) -> str:
        """Generate unique score ID"""
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_hash = hashlib.md5(str(trace).encode()).hexdigest()[:8]
        return f"score_{timestamp}_{trace_hash}"
        
    def _evaluate_metrics(self,
                         reasoning_trace: List[Dict[str, Any]],
                         context: Dict[str, Any]) -> ReasoningMetrics:
        """Evaluate reasoning metrics"""
        metrics = ReasoningMetrics()
        
        # Evaluate clarity
        metrics.clarity = self._evaluate_clarity(reasoning_trace)
        
        # Evaluate relevance
        metrics.relevance = self._evaluate_relevance(reasoning_trace, context)
        
        # Evaluate value
        metrics.value = self._evaluate_value(reasoning_trace, context)
        
        # Evaluate coherence
        metrics.coherence = self._evaluate_coherence(reasoning_trace)
        
        # Evaluate adaptability
        metrics.adaptability = self._evaluate_adaptability(reasoning_trace)
        
        return metrics
        
    def _evaluate_decision_metrics(self,
                                 decision: Dict[str, Any],
                                 context: Dict[str, Any],
                                 expected_outcomes: List[str]) -> ReasoningMetrics:
        """Evaluate decision metrics"""
        metrics = ReasoningMetrics()
        
        # Evaluate clarity
        metrics.clarity = self._evaluate_decision_clarity(decision)
        
        # Evaluate relevance
        metrics.relevance = self._evaluate_decision_relevance(decision, context)
        
        # Evaluate value
        metrics.value = self._evaluate_decision_value(
            decision, context, expected_outcomes)
        
        # Evaluate coherence
        metrics.coherence = self._evaluate_decision_coherence(decision)
        
        # Evaluate adaptability
        metrics.adaptability = self._evaluate_decision_adaptability(decision)
        
        return metrics
        
    def _evaluate_clarity(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate reasoning clarity"""
        if not trace:
            return 0.0
            
        factors = [
            bool(step.get('rationale')),
            bool(step.get('context')),
            bool(step.get('analysis'))
        ]
        return sum(1 for f in factors if f) / len(factors)
        
    def _evaluate_relevance(self,
                          trace: List[Dict[str, Any]],
                          context: Dict[str, Any]) -> float:
        """Evaluate reasoning relevance"""
        if not trace or not context:
            return 0.0
            
        # Check context alignment
        context_keys = set(context.keys())
        trace_keys = set()
        for step in trace:
            trace_keys.update(step.keys())
            
        overlap = len(context_keys.intersection(trace_keys))
        return overlap / len(context_keys) if context_keys else 0.0
        
    def _evaluate_value(self,
                       trace: List[Dict[str, Any]],
                       context: Dict[str, Any]) -> float:
        """Evaluate reasoning value"""
        if not trace:
            return 0.0
            
        # Implementation would assess potential value/utility
        return 0.75  # Placeholder implementation
        
    def _evaluate_coherence(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate reasoning coherence"""
        if not trace:
            return 0.0
            
        # Check step dependencies
        valid_steps = 0
        for i, step in enumerate(trace[1:], 1):
            prev_step = trace[i-1]
            if prev_step.get('id') in str(step.get('dependencies', [])):
                valid_steps += 1
                
        return valid_steps / (len(trace) - 1) if len(trace) > 1 else 1.0
        
    def _evaluate_adaptability(self, trace: List[Dict[str, Any]]) -> float:
        """Evaluate reasoning adaptability"""
        if not trace:
            return 0.0
            
        # Implementation would assess flexibility
        return 0.7  # Placeholder implementation
        
    def _evaluate_decision_clarity(self, decision: Dict[str, Any]) -> float:
        """Evaluate decision clarity"""
        required_fields = ['action', 'parameters', 'rationale']
        return sum(
            1 for field in required_fields
            if field in decision
        ) / len(required_fields)
        
    def _evaluate_decision_relevance(self,
                                   decision: Dict[str, Any],
                                   context: Dict[str, Any]) -> float:
        """Evaluate decision relevance"""
        if not decision or not context:
            return 0.0
            
        # Check context alignment
        context_factors = set(context.keys())
        decision_factors = set(decision.keys())
        
        overlap = len(context_factors.intersection(decision_factors))
        return overlap / len(context_factors) if context_factors else 0.0
        
    def _evaluate_decision_value(self,
                               decision: Dict[str, Any],
                               context: Dict[str, Any],
                               expected_outcomes: List[str]) -> float:
        """Evaluate decision value"""
        if not decision or not expected_outcomes:
            return 0.0
            
        # Check outcome alignment
        addressed_outcomes = sum(
            outcome in str(decision)
            for outcome in expected_outcomes
        )
        return addressed_outcomes / len(expected_outcomes)
        
    def _evaluate_decision_coherence(self, decision: Dict[str, Any]) -> float:
        """Evaluate decision coherence"""
        if not decision:
            return 0.0
            
        # Check internal consistency
        return 0.8  # Placeholder implementation
        
    def _evaluate_decision_adaptability(self, decision: Dict[str, Any]) -> float:
        """Evaluate decision adaptability"""
        if not decision:
            return 0.0
            
        # Check flexibility factors
        adaptability_factors = [
            'alternatives' in decision,
            'fallback' in decision,
            'conditions' in decision
        ]
        return sum(1 for f in adaptability_factors if f) / len(adaptability_factors)
        
    def _analyze_details(self,
                        trace: List[Dict[str, Any]],
                        metrics: ReasoningMetrics) -> Dict[str, Any]:
        """Analyze reasoning details"""
        details = {
            'strengths': [],
            'weaknesses': [],
            'improvement_areas': []
        }
        
        # Analyze clarity
        if metrics.clarity >= 0.8:
            details['strengths'].append("Clear and well-structured reasoning")
        elif metrics.clarity < 0.6:
            details['weaknesses'].append("Reasoning lacks clarity")
            details['improvement_areas'].append("Add more detailed explanations")
            
        # Analyze relevance
        if metrics.relevance >= 0.8:
            details['strengths'].append("Highly relevant to context")
        elif metrics.relevance < 0.6:
            details['weaknesses'].append("Low context relevance")
            details['improvement_areas'].append("Better align with context")
            
        # Analyze value
        if metrics.value >= 0.8:
            details['strengths'].append("High potential value")
        elif metrics.value < 0.6:
            details['weaknesses'].append("Limited value proposition")
            details['improvement_areas'].append("Focus on value generation")
            
        return details
        
    def _analyze_decision_details(self,
                                decision: Dict[str, Any],
                                expected_outcomes: List[str]) -> Dict[str, Any]:
        """Analyze decision details"""
        details = {
            'strengths': [],
            'weaknesses': [],
            'improvement_areas': []
        }
        
        # Check completeness
        if all(outcome in str(decision) for outcome in expected_outcomes):
            details['strengths'].append("Addresses all expected outcomes")
        else:
            details['weaknesses'].append("Missing some expected outcomes")
            details['improvement_areas'].append("Address all expected outcomes")
            
        # Check structure
        if all(field in decision for field in ['action', 'parameters', 'rationale']):
            details['strengths'].append("Well-structured decision")
        else:
            details['weaknesses'].append("Incomplete decision structure")
            details['improvement_areas'].append("Complete all decision fields")
            
        return details
        
    def _calculate_confidence(self,
                            metrics: ReasoningMetrics,
                            details: Dict[str, Any]) -> float:
        """Calculate confidence in evaluation"""
        # Base confidence on metrics
        base_confidence = metrics.get_overall_score()
        
        # Adjust for analysis completeness
        analysis_factor = sum(
            len(details[key]) > 0
            for key in ['strengths', 'weaknesses', 'improvement_areas']
        ) / 3
        
        return (base_confidence * 0.7 + analysis_factor * 0.3)
        
    def _log_score(self,
                   score: ReasoningScore,
                   trace: List[Dict[str, Any]]):
        """Log score details"""
        self.score_history.append({
            'timestamp': datetime.now(),
            'score_id': score.id,
            'target_id': score.target_id,
            'metrics': score.metrics.__dict__,
            'details': score.details,
            'confidence': score.confidence,
            'trace_length': len(trace)
        })
        
    def get_score_history(self,
                         target_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get score history for target"""
        if not target_id:
            return self.score_history
            
        return [
            score for score in self.score_history
            if score['target_id'] == target_id
        ] 