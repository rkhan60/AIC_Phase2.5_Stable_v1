from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum
from .memory_system import MemoryItem, MemoryType, EmotionalTag
import logging

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    STRUCTURAL = "structural"

@dataclass
class ContextTrigger:
    """Definition of a context trigger"""
    trigger_type: TriggerType
    pattern: str
    conditions: List[str]
    activation_threshold: float
    metadata: Dict[str, Any]
    last_activated: Optional[datetime] = None
    activation_count: int = 0

class ContextMatcher:
    """Match contexts using semantic and structural patterns"""
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.context_cache: Dict[str, np.ndarray] = {}
        
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute semantic embedding for text"""
        if text in self.context_cache:
            return self.context_cache[text]
            
        embedding = self.embedding_model.encode([text])[0]
        self.context_cache[text] = embedding
        return embedding
        
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        emb1 = self.compute_embedding(text1)
        emb2 = self.compute_embedding(text2)
        return float(cosine_similarity([emb1], [emb2])[0][0])

class ContextualTrigger:
    """Represents a trigger for context activation"""
    def __init__(self,
                 pattern: str,
                 trigger_type: TriggerType,
                 condition: Any,
                 threshold: float = 0.5):
        self.pattern = pattern
        self.trigger_type = trigger_type
        self.condition = condition
        self.threshold = threshold
        self.last_triggered = None
        self.trigger_count = 0
        
    def check_condition(self, input_data: Any) -> bool:
        """Check if trigger condition is met"""
        # Implementation would vary based on trigger type
        if self.trigger_type == TriggerType.SEMANTIC:
            # Semantic matching would go here
            similarity = 0.8  # Placeholder
            return similarity >= self.threshold
        return False
        
    def update_trigger(self):
        """Update trigger metadata"""
        self.last_triggered = datetime.now()
        self.trigger_count += 1

class ContextualTriggerSystem:
    """Manages context triggers and their activation"""
    def __init__(self):
        self.triggers: Dict[str, ContextualTrigger] = {}
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.matcher = ContextMatcher()  # Add matcher instance
        
    def add_trigger(self,
                   name: str,
                   trigger_type: TriggerType,
                   condition: Any,
                   threshold: float = 0.5) -> ContextualTrigger:
        """Add a new trigger to the system"""
        trigger = ContextualTrigger(name, trigger_type, condition, threshold)
        self.triggers[name] = trigger
        return trigger
        
    def remove_trigger(self, name: str):
        """Remove a trigger from the system"""
        if name in self.triggers:
            del self.triggers[name]
            
    def check_triggers(self, input_data: Any) -> List[str]:
        """Check all triggers against input data"""
        activated = []
        for name, trigger in self.triggers.items():
            if trigger.check_condition(input_data):
                trigger.update_trigger()
                activated.append(name)
        return activated
        
    def get_trigger_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all triggers"""
        stats = {}
        for name, trigger in self.triggers.items():
            stats[name] = {
                "type": trigger.trigger_type.value,
                "count": trigger.trigger_count,
                "last_triggered": trigger.last_triggered
            }
        return stats
        
    def evaluate_trigger(self, trigger_name: str, input_data: Any) -> float:
        """Evaluate a specific trigger's relevance to input data"""
        if trigger_name not in self.triggers:
            logger.info(f"Trigger '{trigger_name}' not found")
            return 0.0
            
        trigger = self.triggers[trigger_name]
        
        if trigger.trigger_type == TriggerType.SEMANTIC:
            # Use matcher for semantic similarity
            similarity = self.matcher.semantic_similarity(trigger.pattern, str(input_data))
            logger.info(f"Semantic similarity for trigger '{trigger_name}': {similarity}")
            return similarity
            
        elif trigger.trigger_type == TriggerType.TEMPORAL:
            # Temporal relevance based on recency
            if not trigger.last_triggered:
                logger.info(f"Temporal trigger '{trigger_name}' never triggered")
                return 0.0
            age = (datetime.now() - trigger.last_triggered).total_seconds()
            relevance = 1.0 / (1.0 + age / 86400)  # Decay over days
            logger.info(f"Temporal relevance for trigger '{trigger_name}': {relevance}")
            return relevance
            
        elif trigger.trigger_type == TriggerType.EMOTIONAL:
            # Simple emotional matching
            if isinstance(trigger.condition, list) and isinstance(input_data, list):
                matching = set(trigger.condition) & set(input_data)
                relevance = len(matching) / len(trigger.condition)
                logger.info(f"Emotional relevance for trigger '{trigger_name}': {relevance}")
                return relevance
                
        logger.info(f"No relevance calculation for trigger type {trigger.trigger_type}")
        return 0.0
        
    def find_relevant_contexts(self, input_data: Any, current_context: str) -> List[str]:
        """Find contexts relevant to input data"""
        relevant = []
        for name, trigger in self.triggers.items():
            relevance = self.evaluate_trigger(name, input_data)
            if relevance >= trigger.threshold:
                relevant.append(name)
        return relevant

class ContextAnalyzer:
    """Analyze context patterns and relationships"""
    def __init__(self, trigger_system: ContextualTriggerSystem):
        self.trigger_system = trigger_system
        
    def analyze_context_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in context activation"""
        patterns = {}
        
        for name, trigger in self.trigger_system.triggers.items():
            patterns[name] = {
                'activation_count': trigger.activation_count,
                'last_activated': trigger.last_activated,
                'trigger_type': trigger.trigger_type.value,
                'conditions': trigger.conditions
            }
            
        return patterns
        
    def find_related_contexts(self, context_name: str) -> List[str]:
        """Find contexts related to given context"""
        if context_name not in self.trigger_system.triggers:
            return []
            
        related = []
        target_trigger = self.trigger_system.triggers[context_name]
        
        for name, trigger in self.trigger_system.triggers.items():
            if name == context_name:
                continue
                
            # Check for shared conditions
            shared_conditions = set(target_trigger.conditions) & set(trigger.conditions)
            if shared_conditions:
                related.append(name)
                
            # Check for semantic similarity in patterns
            if self.trigger_system.matcher.semantic_similarity(
                target_trigger.pattern, trigger.pattern) > 0.8:
                related.append(name)
                
        return list(set(related))
        
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about context usage"""
        stats = {
            'total_triggers': len(self.trigger_system.triggers),
            'active_contexts': len(self.trigger_system.active_contexts),
            'trigger_types': {
                trigger_type: len([t for t in self.trigger_system.triggers.values()
                                 if t.trigger_type == trigger_type])
                for trigger_type in TriggerType
            },
            'most_active': sorted(
                self.trigger_system.triggers.items(),
                key=lambda x: x[1].activation_count,
                reverse=True
            )[:5]
        }
        return stats

class ContextBuilder:
    """Helper for building context triggers"""
    @staticmethod
    def create_semantic_trigger(pattern: str,
                              threshold: float = 0.7) -> ContextTrigger:
        """Create semantic matching trigger"""
        return ContextTrigger(
            trigger_type=TriggerType.SEMANTIC,
            pattern=pattern,
            conditions=[],
            activation_threshold=threshold,
            metadata={}
        )
        
    @staticmethod
    def create_structural_trigger(pattern: str,
                                conditions: List[str]) -> ContextTrigger:
        """Create structural pattern trigger"""
        return ContextTrigger(
            trigger_type=TriggerType.STRUCTURAL,
            pattern=pattern,
            conditions=conditions,
            activation_threshold=1.0,
            metadata={}
        )
        
    @staticmethod
    def create_composite_trigger(pattern: str,
                               conditions: List[str],
                               threshold: float = 0.7) -> ContextTrigger:
        """Create composite trigger"""
        return ContextTrigger(
            trigger_type=TriggerType.COMPOSITE,
            pattern=pattern,
            conditions=conditions,
            activation_threshold=threshold,
            metadata={}
        ) 