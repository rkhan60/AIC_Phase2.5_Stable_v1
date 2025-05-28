from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from .memory_system import MemoryItem, MemoryType, EmotionalTag

@dataclass
class DecayConfig:
    """Configuration for memory decay"""
    base_half_life: timedelta = timedelta(days=30)
    min_importance: float = 0.1
    reinforcement_boost: float = 0.2
    decay_rate: float = 0.1
    access_threshold: int = 5
    time_threshold: timedelta = timedelta(days=7)

class DecayFunction:
    """Base class for decay functions"""
    def calculate(self, age: timedelta, importance: float) -> float:
        raise NotImplementedError

class ExponentialDecay(DecayFunction):
    """Exponential decay function"""
    def calculate(self, age: timedelta, importance: float) -> float:
        half_life = timedelta(days=30)
        decay = 2 ** (-age / half_life)
        return max(0.1, decay * importance)

class LinearDecay(DecayFunction):
    """Linear decay function"""
    def calculate(self, age: timedelta, importance: float) -> float:
        max_age = timedelta(days=365)
        decay = 1 - (age / max_age)
        return max(0.1, decay * importance)

class ReinforcementSystem:
    """System for memory reinforcement"""
    def __init__(self, config: DecayConfig = None):
        self.config = config or DecayConfig()
        self.access_history: Dict[str, List[datetime]] = {}
        self.reinforcement_scores: Dict[str, float] = {}

    def record_access(self, memory_id: str, timestamp: datetime = None):
        """Record memory access"""
        timestamp = timestamp or datetime.now()
        if memory_id not in self.access_history:
            self.access_history[memory_id] = []
        self.access_history[memory_id].append(timestamp)

    def calculate_reinforcement(self, memory_id: str) -> float:
        """Calculate reinforcement score based on access patterns"""
        if memory_id not in self.access_history:
            return 0.0

        accesses = self.access_history[memory_id]
        recent_accesses = [
            ts for ts in accesses
            if datetime.now() - ts <= self.config.time_threshold
        ]

        if len(recent_accesses) >= self.config.access_threshold:
            return self.config.reinforcement_boost

        # Calculate frequency-based score
        frequency_score = len(recent_accesses) / self.config.access_threshold
        return frequency_score * self.config.reinforcement_boost

class DecayManager:
    """Manager for memory decay and reinforcement"""
    def __init__(self, config: DecayConfig = None):
        self.config = config or DecayConfig()
        self.reinforcement = ReinforcementSystem(config)
        self.decay_functions: Dict[MemoryType, DecayFunction] = {
            MemoryType.WORKING: LinearDecay(),
            MemoryType.LONG_TERM: ExponentialDecay(),
            MemoryType.EMOTIONAL: ExponentialDecay()
        }

    def process_memory(self, memory: MemoryItem, memory_id: str) -> MemoryItem:
        """Process memory decay and reinforcement"""
        # Calculate base decay
        age = datetime.now() - memory.created_at
        decay_func = self.decay_functions[memory.memory_type]
        decayed_importance = decay_func.calculate(age, memory.importance_score)

        # Apply reinforcement
        reinforcement = self.reinforcement.calculate_reinforcement(memory_id)
        final_importance = min(1.0, decayed_importance + reinforcement)

        # Update memory
        memory.importance_score = final_importance
        return memory

    def update_access(self, memory_id: str, memory: MemoryItem):
        """Update access records for reinforcement"""
        self.reinforcement.record_access(memory_id)
        memory.access_count += 1
        memory.last_accessed = datetime.now()

class MemoryOptimizer:
    """Optimize memory storage and retrieval"""
    def __init__(self, decay_manager: DecayManager):
        self.decay_manager = decay_manager
        self.importance_threshold = 0.3
        self.consolidation_threshold = 0.7

    def should_retain(self, memory: MemoryItem) -> bool:
        """Determine if memory should be retained"""
        return memory.importance_score >= self.importance_threshold

    def should_consolidate(self, memory: MemoryItem) -> bool:
        """Determine if memory should be consolidated"""
        return memory.importance_score >= self.consolidation_threshold

    def optimize_memory(self, memory: MemoryItem, memory_id: str) -> Optional[MemoryItem]:
        """Optimize memory based on importance and access patterns"""
        # Process decay and reinforcement
        processed_memory = self.decay_manager.process_memory(memory, memory_id)

        # Check retention threshold
        if not self.should_retain(processed_memory):
            return None

        # Apply memory type transitions
        if self.should_consolidate(processed_memory):
            if processed_memory.memory_type == MemoryType.WORKING:
                processed_memory.memory_type = MemoryType.LONG_TERM

        return processed_memory

class DecayAnalytics:
    """Analytics for decay and reinforcement patterns"""
    def __init__(self, decay_manager: DecayManager):
        self.decay_manager = decay_manager

    def analyze_decay_patterns(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Analyze decay patterns across memories"""
        patterns = {
            'avg_importance': np.mean([m.importance_score for m in memories]),
            'decay_distribution': {
                'high': len([m for m in memories if m.importance_score >= 0.7]),
                'medium': len([m for m in memories if 0.3 <= m.importance_score < 0.7]),
                'low': len([m for m in memories if m.importance_score < 0.3])
            },
            'memory_type_distribution': {
                memory_type: len([m for m in memories if m.memory_type == memory_type])
                for memory_type in MemoryType
            }
        }
        return patterns

    def analyze_reinforcement_patterns(self) -> Dict[str, Any]:
        """Analyze reinforcement patterns"""
        access_patterns = {
            memory_id: {
                'access_count': len(accesses),
                'last_access': max(accesses) if accesses else None,
                'reinforcement_score': self.decay_manager.reinforcement.calculate_reinforcement(memory_id)
            }
            for memory_id, accesses in self.decay_manager.reinforcement.access_history.items()
        }
        return access_patterns 