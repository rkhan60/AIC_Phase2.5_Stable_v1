from typing import List, Dict, Any, Set, Optional
from datetime import datetime, timedelta
from .memory_system import MemoryItem, MemoryType
from .context_system import ContextualTriggerSystem
from .hierarchical_context import HierarchicalContextSystem
import logging

logger = logging.getLogger(__name__)

class ContextAwareConsolidation:
    """Manages context-aware memory consolidation"""
    def __init__(self,
                 trigger_system: ContextualTriggerSystem,
                 hierarchy: HierarchicalContextSystem):
        self.trigger_system = trigger_system
        self.hierarchy = hierarchy
        self.consolidation_threshold = 0.5  # Lower threshold for more frequent consolidation
        self.consolidation_history: List[Dict[str, Any]] = []
        
    def consolidate_memory(self,
                         memory: MemoryItem,
                         current_context: str) -> Optional[MemoryItem]:
        """Consolidate memory based on context"""
        if not self._should_consolidate(memory, current_context):
            return None
            
        # Create consolidated memory
        consolidated = MemoryItem(
            content=memory.content,
            memory_type=memory.memory_type,
            context_tags=self._get_consolidated_tags(memory, current_context),
            emotional_tags=memory.emotional_tags,
            importance_score=self._calculate_consolidated_importance(
                memory, current_context),
            last_accessed=datetime.now()
        )
        
        # Record consolidation
        self._record_consolidation(memory, consolidated, current_context)
        
        return consolidated
        
    def _should_consolidate(self,
                          memory: MemoryItem,
                          current_context: str) -> bool:
        """Determine if memory should be consolidated"""
        # Check importance threshold
        if memory.importance_score < self.consolidation_threshold:
            return False
            
        # Check context relevance
        relevance = self._calculate_context_relevance(memory, current_context)
        if relevance < self.consolidation_threshold:
            return False
            
        # Check temporal factors
        if memory.last_accessed:
            age = datetime.now() - memory.last_accessed
            if age < timedelta(minutes=30):  # Too recent
                return False
                
        return True
        
    def _calculate_context_relevance(self,
                                   memory: MemoryItem,
                                   current_context: str) -> float:
        """Calculate relevance to current context"""
        # Check direct context match
        if current_context in memory.context_tags:
            return 1.0
            
        # Check hierarchical relevance
        if current_context in self.hierarchy.nodes:
            node = self.hierarchy.nodes[current_context]
            ancestors = node.get_ancestors()
            for ancestor in ancestors:
                if ancestor.name in memory.context_tags:
                    return 0.8
                    
        # Check trigger activation
        activated = self.trigger_system.check_triggers(memory.content)
        if current_context in activated:
            return 0.9
            
        return 0.5
        
    def _get_consolidated_tags(self,
                             memory: MemoryItem,
                             current_context: str) -> List[str]:
        """Get consolidated context tags"""
        tags = set(memory.context_tags)
        
        # Add current context if relevant
        if self._calculate_context_relevance(memory, current_context) > 0.7:
            tags.add(current_context)
            
        # Add relevant ancestor contexts
        if current_context in self.hierarchy.nodes:
            node = self.hierarchy.nodes[current_context]
            for ancestor in node.get_ancestors():
                if self._calculate_context_relevance(memory, ancestor.name) > 0.5:
                    tags.add(ancestor.name)
                    
        return list(tags)
        
    def _calculate_consolidated_importance(self,
                                        memory: MemoryItem,
                                        current_context: str) -> float:
        """Calculate consolidated importance score"""
        base_score = memory.importance_score
        
        # Adjust based on context relevance
        relevance = self._calculate_context_relevance(memory, current_context)
        score = base_score * (1 + relevance) / 2
        
        # Adjust based on emotional impact
        emotional_factor = len(memory.emotional_tags) * 0.1
        score = min(1.0, score + emotional_factor)
        
        # Adjust based on temporal factors
        if memory.last_accessed:
            age = datetime.now() - memory.last_accessed
            recency_factor = 1.0 / (1.0 + age.total_seconds() / 86400)  # Decay over days
            score = score * (0.7 + 0.3 * recency_factor)
            
        return score
        
    def _record_consolidation(self,
                            original: MemoryItem,
                            consolidated: MemoryItem,
                            context: str):
        """Record consolidation event"""
        self.consolidation_history.append({
            "timestamp": datetime.now(),
            "context": context,
            "original_score": original.importance_score,
            "consolidated_score": consolidated.importance_score,
            "context_tags": consolidated.context_tags
        })
        
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics"""
        if not self.consolidation_history:
            return {}
            
        stats = {
            "total_consolidations": len(self.consolidation_history),
            "average_score_increase": 0.0,
            "context_distribution": {},
            "last_consolidation": self.consolidation_history[-1]["timestamp"]
        }
        
        score_increases = []
        for event in self.consolidation_history:
            increase = event["consolidated_score"] - event["original_score"]
            score_increases.append(increase)
            
            context = event["context"]
            stats["context_distribution"][context] = \
                stats["context_distribution"].get(context, 0) + 1
                
        if score_increases:
            stats["average_score_increase"] = sum(score_increases) / len(score_increases)
            
        return stats
        
    def evaluate_consolidation_priority(self,
                                     memory: MemoryItem,
                                     current_context: str) -> float:
        """Calculate consolidation priority based on context relevance"""
        # Get relevant contexts
        contexts = self.trigger_system.find_relevant_contexts(memory, current_context)
        
        if not contexts:
            return 0.0
            
        # Calculate priority based on multiple factors
        priority = 0.0
        active_contexts = self.hierarchy.get_active_contexts()
        
        # Context relevance (40%)
        context_score = 0.0
        for context_name in contexts:
            relevance = self.trigger_system.evaluate_trigger(context_name, str(memory.content))
            context_score = max(context_score, relevance)
        priority += 0.4 * context_score
        
        # Temporal relevance (30%)
        if memory.last_accessed:
            age = (datetime.now() - memory.last_accessed).total_seconds() / 3600
            temporal_score = 1.0 if age < 24 else (1.0 / (1.0 + (age - 24) / 24))
            priority += 0.3 * temporal_score
            
        # Emotional relevance (30%)
        emotional_score = 0.0
        if memory.emotional_tags:
            emotional_score = len(memory.emotional_tags) / 3  # Normalize by max expected tags
        priority += 0.3 * min(1.0, emotional_score)
        
        return min(1.0, priority)
        
    def should_consolidate(self,
                          memory: MemoryItem,
                          current_context: str) -> bool:
        """Determine if memory should be consolidated based on context"""
        priority = self.evaluate_consolidation_priority(memory, current_context)
        threshold = self.consolidation_threshold
        return priority >= threshold
        
    def get_consolidation_contexts(self,
                                 memory: MemoryItem,
                                 current_context: str) -> Set[str]:
        """Get relevant contexts for memory consolidation"""
        contexts = set()
        
        # Get directly relevant contexts
        direct_contexts = self.trigger_system.find_relevant_contexts(
            memory, current_context)
        contexts.update(direct_contexts)
        
        # Get hierarchically related contexts
        for context_name in direct_contexts:
            node = self.hierarchy.nodes.get(context_name)
            if not node:
                continue
                
            # Add ancestor contexts with decreasing relevance
            ancestors = node.get_ancestors()
            for ancestor in ancestors:
                if self.evaluate_context_relevance(memory, ancestor.name) > 0.5:
                    contexts.add(ancestor.name)
                    
            # Add child contexts that are highly relevant
            for child in node.children:
                if self.evaluate_context_relevance(memory, child.name) > 0.7:
                    contexts.add(child.name)
                    
        return contexts
        
    def evaluate_context_relevance(self,
                                 memory: MemoryItem,
                                 context_name: str) -> float:
        """Evaluate how relevant a context is to a memory"""
        node = self.hierarchy.nodes.get(context_name)
        if not node:
            return 0.0
            
        relevance = 0.0
        
        # Check trigger conditions
        trigger_relevance = self.trigger_system.evaluate_trigger(
            context_name, str(memory.content))
        relevance = max(relevance, trigger_relevance)
            
        # Check metadata matching
        if any(tag in memory.context_tags for tag in node.metadata.get("tags", [])):
            relevance += 0.2
            
        # Check temporal relevance
        if node.last_activated and memory.last_accessed:
            time_diff = abs((node.last_activated - memory.last_accessed).total_seconds())
            if time_diff < 3600:  # Within 1 hour
                relevance += 0.2
                
        return min(1.0, relevance)
        
    def update_memory_contexts(self,
                             memory: MemoryItem,
                             current_context: str):
        """Update memory's context tags based on current context"""
        relevant_contexts = self.get_consolidation_contexts(memory, current_context)
        
        # Update context tags
        memory.context_tags = list(set(memory.context_tags) | relevant_contexts)
        
        # Update context activation
        for context_name in relevant_contexts:
            self.hierarchy.update_context_activation(context_name)
            
    def consolidate_memory(self,
                          memory: MemoryItem,
                          current_context: str) -> Optional[MemoryItem]:
        """Consolidate memory with context awareness"""
        priority = self.evaluate_consolidation_priority(memory, current_context)
        logger.info(f"Consolidation priority for context '{current_context}': {priority}")
        
        if not self.should_consolidate(memory, current_context):
            logger.info(f"Memory not consolidated: priority {priority} < threshold {self.consolidation_threshold}")
            return None
            
        # Update memory's context information
        self.update_memory_contexts(memory, current_context)
        
        # Adjust memory importance based on context relevance
        memory.importance_score = max(memory.importance_score, priority)
        
        # Update last accessed time
        memory.last_accessed = datetime.now()
        
        logger.info(f"Memory consolidated with importance score: {memory.importance_score}")
        return memory
        
    def get_related_memories(self,
                           context_name: str,
                           memories: List[MemoryItem],
                           threshold: float = 0.5) -> List[MemoryItem]:
        """Find memories related to a context"""
        related = []
        
        for memory in memories:
            relevance = self.evaluate_context_relevance(memory, context_name)
            if relevance >= threshold:
                related.append(memory)
                
        # Sort by relevance and recency
        related.sort(key=lambda m: (
            self.evaluate_context_relevance(m, context_name),
            m.last_accessed or datetime.min
        ), reverse=True)
        
        return related 