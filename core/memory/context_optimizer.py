from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from .memory_system import MemoryItem
from .hierarchical_context import HierarchicalContextSystem
from .context_aware_consolidation import ContextAwareConsolidation

class CacheEntry:
    """Represents a cached list of memories with metadata"""
    def __init__(self, memories: List[MemoryItem], context: str):
        self.memories = memories
        self.context = context
        self.cache_time = datetime.now()
        self.access_count = 0
        self.last_access = None
        self.metadata = {
            'size': len(memories),
            'contexts': set(m.context_tags for m in memories),
            'importance_range': (
                min(m.importance_score for m in memories),
                max(m.importance_score for m in memories)
            ) if memories else (0.0, 0.0)
        }
        
    def is_valid(self, ttl: timedelta) -> bool:
        """Check if cache entry is still valid"""
        if not self.cache_time:
            return False
        age = datetime.now() - self.cache_time
        return age <= ttl
        
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = datetime.now()
        
    def should_refresh(self, max_age: timedelta, max_accesses: int = 100) -> bool:
        """Determine if entry should be refreshed"""
        if not self.is_valid(max_age):
            return True
        if self.access_count >= max_accesses:
            return True
        return False

class MemoryOptimizationMetrics:
    """Tracks optimization metrics"""
    def __init__(self):
        self.access_patterns: Dict[str, int] = defaultdict(int)
        self.context_switches: Dict[str, int] = defaultdict(int)
        self.retrieval_times: Dict[str, List[float]] = defaultdict(list)
        self.consolidation_stats: Dict[str, int] = defaultdict(int)
        
    def update_access(self, context: str):
        """Update access count for context"""
        self.access_patterns[context] += 1
        
    def record_context_switch(self, from_context: str, to_context: str):
        """Record context switch"""
        key = f"{from_context}->{to_context}"
        self.context_switches[key] += 1
        
    def record_retrieval_time(self, context: str, time_ms: float):
        """Record memory retrieval time"""
        self.retrieval_times[context].append(time_ms)
        
    def record_consolidation(self, context: str):
        """Record memory consolidation"""
        self.consolidation_stats[context] += 1
        
    def get_average_retrieval_time(self, context: str) -> float:
        """Get average retrieval time for context"""
        times = self.retrieval_times.get(context, [])
        return np.mean(times) if times else 0.0
        
    def get_most_accessed_contexts(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most frequently accessed contexts"""
        sorted_contexts = sorted(
            self.access_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_contexts[:limit]
        
    def get_common_transitions(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most common context transitions"""
        sorted_transitions = sorted(
            self.context_switches.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_transitions[:limit]

class ContextOptimizer:
    """Optimizes memory operations based on context patterns"""
    def __init__(self,
                 hierarchy: HierarchicalContextSystem,
                 consolidation: ContextAwareConsolidation):
        self.hierarchy = hierarchy
        self.consolidation = consolidation
        self.metrics = MemoryOptimizationMetrics()
        self.context_cache: Dict[str, CacheEntry] = {}
        self.cache_size = 100
        self.cache_ttl = timedelta(hours=1)
        self.max_cache_accesses = 100
        self.last_context = None
        
    def optimize_retrieval(self,
                          memories: List[MemoryItem],
                          context: str) -> List[MemoryItem]:
        """Optimize memory retrieval for context"""
        start_time = datetime.now()
        
        # Record context switch if applicable
        if self.last_context and self.last_context != context:
            self.metrics.record_context_switch(self.last_context, context)
        self.last_context = context
        
        # Check cache first
        if self.check_cache(context):
            cached = self.context_cache[context].memories
            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.record_retrieval_time(context, retrieval_time)
            return cached
            
        # Get relevant memories
        relevant = self.consolidation.get_related_memories(context, memories)
        
        # Update cache
        self.update_cache(context, relevant)
        
        # Update metrics
        self.metrics.update_access(context)
        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.record_retrieval_time(context, retrieval_time)
        
        return relevant
        
    def check_cache(self, context: str) -> bool:
        """Check if context is in cache and valid"""
        if context not in self.context_cache:
            return False
            
        entry = self.context_cache[context]
        if not entry.is_valid(self.cache_ttl) or entry.should_refresh(self.cache_ttl, self.max_cache_accesses):
            del self.context_cache[context]
            return False
            
        entry.update_access()
        return True
        
    def update_cache(self, context: str, memories: List[MemoryItem]):
        """Update context cache"""
        # Remove old entries if cache is full
        while len(self.context_cache) >= self.cache_size:
            # Remove least recently accessed entry
            oldest_context = min(
                self.context_cache.keys(),
                key=lambda k: self.context_cache[k].last_access or datetime.min
            )
            del self.context_cache[oldest_context]
            
        # Add new entry
        self.context_cache[context] = CacheEntry(memories, context)
        
    def predict_next_contexts(self, current_context: str) -> List[str]:
        """Predict likely next contexts based on transition patterns"""
        transitions = defaultdict(int)
        
        # Analyze transitions from current context
        for transition, count in self.metrics.context_switches.items():
            from_context, to_context = transition.split("->")
            if from_context == current_context:
                transitions[to_context] += count
                
        # Sort by frequency
        predicted = sorted(
            transitions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [context for context, _ in predicted[:3]]
        
    def optimize_consolidation(self,
                             memory: MemoryItem,
                             current_context: str) -> Optional[MemoryItem]:
        """Optimize memory consolidation"""
        # Check if consolidation is needed based on context patterns
        if not self._should_optimize_consolidation(memory, current_context):
            return None
            
        # Perform consolidation
        consolidated = self.consolidation.consolidate_memory(
            memory, current_context)
            
        if consolidated:
            self.metrics.record_consolidation(current_context)
            
        return consolidated
        
    def _should_optimize_consolidation(self,
                                     memory: MemoryItem,
                                     context: str) -> bool:
        """Determine if consolidation should be optimized"""
        # Check access frequency
        access_count = self.metrics.access_patterns.get(context, 0)
        if access_count > 10:  # High access context
            return True
            
        # Check transition patterns
        transitions = self.predict_next_contexts(context)
        if transitions:  # Context has predictable transitions
            return True
            
        # Check retrieval performance
        avg_time = self.metrics.get_average_retrieval_time(context)
        if avg_time > 100:  # Slow retrieval
            return True
            
        return False
        
    def get_optimization_report(self) -> Dict[str, any]:
        """Generate optimization performance report"""
        report = {
            "most_accessed_contexts": self.metrics.get_most_accessed_contexts(),
            "common_transitions": self.metrics.get_common_transitions(),
            "retrieval_times": {
                context: {
                    "average": np.mean(times),
                    "min": np.min(times),
                    "max": np.max(times)
                }
                for context, times in self.metrics.retrieval_times.items()
            },
            "consolidation_stats": dict(self.metrics.consolidation_stats),
            "cache_stats": {
                "size": len(self.context_cache),
                "contexts": list(self.context_cache.keys())
            }
        }
        return report
        
    def optimize_context_structure(self) -> List[str]:
        """Optimize context hierarchy based on usage patterns"""
        recommendations = []
        
        # Analyze access patterns
        frequent_contexts = self.metrics.get_most_accessed_contexts()
        for context, count in frequent_contexts:
            node = self.hierarchy.nodes.get(context)
            if not node:
                continue
                
            # Check if context should be moved up in hierarchy
            if count > 100 and len(node.get_ancestors()) > 2:
                recommendations.append(
                    f"Consider moving '{context}' up in hierarchy for faster access"
                )
                
            # Check if context should be split
            if len(node.children) > 5:
                recommendations.append(
                    f"Consider splitting '{context}' into multiple contexts"
                )
                
        # Analyze transition patterns
        transitions = self.metrics.get_common_transitions()
        for transition, count in transitions:
            from_context, to_context = transition.split("->")
            if count > 50:
                recommendations.append(
                    f"Consider creating a direct relationship between "
                    f"'{from_context}' and '{to_context}'"
                )
                
        return recommendations 