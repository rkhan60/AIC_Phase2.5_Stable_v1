from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import logging
from ..memory.memory_query import MemoryQueryEngine
from ..memory.memory_system import MemoryItem
from ..memory.context_system import ContextualTriggerSystem

logger = logging.getLogger(__name__)

class AgentActionType(Enum):
    QUERY = "query"
    ANALYZE = "analyze"
    RECOMMEND = "recommend"
    WARN = "warn"
    EXECUTE = "execute"

class AgentAction:
    """Represents a structured action from an agent"""
    def __init__(self,
                 action_type: AgentActionType,
                 content: Any,
                 confidence: float,
                 metadata: Optional[Dict[str, Any]] = None):
        self.action_type = action_type
        self.content = content
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary"""
        return {
            'type': self.action_type.value,
            'content': self.content,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

class ThoughtAgent:
    """Agent for processing thoughts and generating structured actions"""
    def __init__(self,
                 memory_query: MemoryQueryEngine,
                 trigger_system: Optional[ContextualTriggerSystem] = None):
        self.memory_query = memory_query
        self.trigger_system = trigger_system
        self.action_history: List[AgentAction] = []
        self.logs: List[Dict[str, Any]] = []
        
    def process_input(self,
                     input_data: Any,
                     context: str,
                     memories: List[MemoryItem]) -> List[AgentAction]:
        """Process input and generate appropriate actions"""
        actions = []
        
        # Check for similar past experiences
        similar = self.memory_query.find_similar_experiences(str(input_data), memories)
        if similar:
            action = AgentAction(
                AgentActionType.ANALYZE,
                f"Found {len(similar)} similar past experiences",
                confidence=max(m['similarity'] for m in similar),
                metadata={'similar_experiences': similar}
            )
            actions.append(action)
            
        # Check for potential failures
        failures = self.memory_query.find_failure_patterns(input_data, memories)
        if failures:
            action = AgentAction(
                AgentActionType.WARN,
                "Detected potential failure patterns",
                confidence=max(f['similarity'] for f in failures),
                metadata={'failure_patterns': failures}
            )
            actions.append(action)
            
        # Query successful strategies
        strategies = self.memory_query.query_successful_strategies(context, memories)
        if strategies:
            action = AgentAction(
                AgentActionType.RECOMMEND,
                "Found relevant successful strategies",
                confidence=max(s['similarity'] for s in strategies),
                metadata={'strategies': strategies}
            )
            actions.append(action)
            
        # Log actions
        for action in actions:
            self._log_action(action)
            self.action_history.append(action)
            
        return actions
        
    def get_action_feedback(self,
                          action: AgentAction,
                          feedback: Dict[str, Any]) -> AgentAction:
        """Process feedback for an action"""
        # Create feedback action
        feedback_action = AgentAction(
            action.action_type,
            f"Feedback for {action.action_type.value}",
            confidence=feedback.get('confidence', 0.5),
            metadata={
                'original_action': action.to_dict(),
                'feedback': feedback
            }
        )
        
        self._log_action(feedback_action, is_feedback=True)
        return feedback_action
        
    def _log_action(self, action: AgentAction, is_feedback: bool = False):
        """Log agent action with metadata"""
        self.logs.append({
            'timestamp': datetime.now(),
            'action': action.to_dict(),
            'is_feedback': is_feedback
        })
        
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about agent actions"""
        if not self.action_history:
            return {}
            
        stats = {
            'total_actions': len(self.action_history),
            'action_types': {},
            'avg_confidence': 0.0,
            'last_action': self.action_history[-1].timestamp
        }
        
        # Calculate action type distribution
        for action in self.action_history:
            action_type = action.action_type.value
            stats['action_types'][action_type] = \
                stats['action_types'].get(action_type, 0) + 1
                
        # Calculate average confidence
        stats['avg_confidence'] = sum(
            action.confidence for action in self.action_history
        ) / len(self.action_history)
        
        return stats 