from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import numpy as np

@dataclass
class DecayConfig:
    """Configuration for memory decay"""
    base_decay_rate: float = 0.1  # Base rate at which memory decays
    time_decay_factor: float = 0.05  # How much time affects decay
    min_confidence: float = 0.1  # Minimum confidence threshold
    reinforcement_boost: float = 0.2  # How much reinforcement strengthens memory
    max_confidence: float = 1.0  # Maximum confidence value
    decay_check_interval: timedelta = timedelta(days=1)  # How often to check for decay

class DecayManager:
    """Manages memory decay and reinforcement"""
    
    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
        self.last_decay_check = datetime.now()
        
    def calculate_time_decay(self, created_at: datetime, last_used: datetime) -> float:
        """Calculate decay based on time"""
        now = datetime.now()
        age = (now - created_at).total_seconds()
        time_since_last_use = (now - last_used).total_seconds()
        
        # Exponential decay based on age and time since last use
        age_decay = math.exp(-self.config.time_decay_factor * age / (24 * 3600))  # Convert to days
        usage_decay = math.exp(-self.config.time_decay_factor * time_since_last_use / (24 * 3600))
        
        return (age_decay + usage_decay) / 2

    def apply_decay(self, current_confidence: float, 
                   created_at: datetime,
                   last_used: datetime,
                   use_count: int) -> float:
        """Apply decay to confidence value"""
        # Base decay
        decay_amount = self.config.base_decay_rate
        
        # Time-based decay
        time_factor = self.calculate_time_decay(created_at, last_used)
        decay_amount *= (1 - time_factor)
        
        # Usage frequency adjustment
        usage_factor = 1 / (1 + math.log(use_count + 1))
        decay_amount *= usage_factor
        
        # Apply decay
        new_confidence = current_confidence * (1 - decay_amount)
        
        # Ensure confidence doesn't go below minimum
        return max(new_confidence, self.config.min_confidence)

    def apply_reinforcement(self, current_confidence: float, 
                          success_score: float,
                          use_count: int) -> float:
        """Apply reinforcement to confidence value"""
        # Base reinforcement
        boost = self.config.reinforcement_boost
        
        # Scale boost based on success score
        boost *= success_score
        
        # Adjust boost based on usage frequency
        frequency_factor = math.log(use_count + 1) / 10  # Diminishing returns
        boost *= (1 + frequency_factor)
        
        # Apply reinforcement
        new_confidence = current_confidence + boost
        
        # Ensure confidence doesn't exceed maximum
        return min(new_confidence, self.config.max_confidence)

    def should_prune(self, confidence: float, 
                    last_used: datetime,
                    use_count: int) -> bool:
        """Determine if memory should be pruned"""
        # Check if confidence is too low
        if confidence <= self.config.min_confidence:
            # Allow more time for frequently used memories
            grace_period = timedelta(days=30 * math.log(use_count + 1))
            return datetime.now() - last_used > grace_period
        return False

    def calculate_memory_weight(self, confidence: float,
                              created_at: datetime,
                              last_used: datetime,
                              use_count: int) -> float:
        """Calculate overall memory importance weight"""
        # Time factors
        recency = 1 / (1 + (datetime.now() - last_used).total_seconds() / (24 * 3600))
        age_factor = 1 / (1 + (datetime.now() - created_at).total_seconds() / (24 * 3600))
        
        # Usage factor
        usage_factor = math.log(use_count + 1) / 10
        
        # Combine factors
        weight = confidence * (0.4 * recency + 0.3 * age_factor + 0.3 * usage_factor)
        return weight

    def update_decay_config(self, updates: Dict[str, Any]):
        """Update decay configuration parameters"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def should_check_decay(self) -> bool:
        """Check if it's time to perform decay operations"""
        return datetime.now() - self.last_decay_check >= self.config.decay_check_interval

    def mark_decay_check(self):
        """Mark that decay check has been performed"""
        self.last_decay_check = datetime.now()

    def get_adaptive_parameters(self, memory_stats: Dict[str, Any]) -> Dict[str, float]:
        """Get adaptive decay parameters based on memory system statistics"""
        avg_confidence = memory_stats.get('avg_confidence', 0.5)
        total_memories = memory_stats.get('total_memories', 0)
        avg_age = memory_stats.get('avg_age', 0)
        
        # Adjust decay rate based on system state
        adaptive_params = {
            'base_decay_rate': self._adjust_decay_rate(avg_confidence, total_memories),
            'reinforcement_boost': self._adjust_reinforcement(avg_confidence),
            'time_decay_factor': self._adjust_time_factor(avg_age)
        }
        
        return adaptive_params
        
    def _adjust_decay_rate(self, avg_confidence: float, total_memories: int) -> float:
        """Adjust base decay rate based on system state"""
        # Increase decay if average confidence is too high
        confidence_factor = 1 + max(0, (avg_confidence - 0.7) * 2)
        
        # Increase decay if too many memories
        memory_factor = 1 + max(0, (total_memories - 1000) / 1000)
        
        return self.config.base_decay_rate * confidence_factor * memory_factor
        
    def _adjust_reinforcement(self, avg_confidence: float) -> float:
        """Adjust reinforcement boost based on average confidence"""
        if avg_confidence < 0.3:
            # Increase reinforcement for low confidence
            return self.config.reinforcement_boost * 1.5
        elif avg_confidence > 0.8:
            # Decrease reinforcement for high confidence
            return self.config.reinforcement_boost * 0.7
        return self.config.reinforcement_boost
        
    def _adjust_time_factor(self, avg_age: float) -> float:
        """Adjust time decay factor based on average memory age"""
        # Increase time factor for older memory sets
        if avg_age > 30:  # More than 30 days
            return self.config.time_decay_factor * 1.3
        return self.config.time_decay_factor 