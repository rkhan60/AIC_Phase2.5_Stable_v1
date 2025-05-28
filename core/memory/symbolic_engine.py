from typing import Dict, List, Any, Optional, Callable, Pattern
import re
from dataclasses import dataclass
from enum import Enum
from .memory_system import MemoryItem, MemoryType, EmotionalTag

class RuleType(Enum):
    PATTERN = "pattern"
    TRANSFORMATION = "transformation"
    INFERENCE = "inference"
    EMOTIONAL = "emotional"

@dataclass
class SymbolicRule:
    """Definition of a symbolic rule"""
    rule_type: RuleType
    pattern: Pattern  # Regex pattern
    action: Callable
    priority: int
    conditions: List[Callable]
    metadata: Dict[str, Any]

class PatternMatcher:
    """Pattern matching system for memory content"""
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        
    def add_pattern(self, name: str, pattern: str):
        """Add new pattern to the system"""
        self.patterns[name] = re.compile(pattern)
        
    def match(self, content: str) -> List[str]:
        """Find all matching patterns"""
        matches = []
        for name, pattern in self.patterns.items():
            if pattern.search(content):
                matches.append(name)
        return matches

class SymbolicEngine:
    """Core symbolic reasoning engine"""
    def __init__(self):
        self.rules: Dict[str, SymbolicRule] = {}
        self.pattern_matcher = PatternMatcher()
        self.transformation_cache: Dict[str, Any] = {}
        
    def add_rule(self, 
                 name: str,
                 rule_type: RuleType,
                 pattern: str,
                 action: Callable,
                 priority: int = 0,
                 conditions: List[Callable] = None,
                 metadata: Dict[str, Any] = None):
        """Add new symbolic rule"""
        self.rules[name] = SymbolicRule(
            rule_type=rule_type,
            pattern=re.compile(pattern),
            action=action,
            priority=priority,
            conditions=conditions or [],
            metadata=metadata or {}
        )
        
        if rule_type == RuleType.PATTERN:
            self.pattern_matcher.add_pattern(name, pattern)
            
    def process_memory(self, memory: MemoryItem) -> Optional[MemoryItem]:
        """Process memory through symbolic rules"""
        # Sort rules by priority
        sorted_rules = sorted(
            self.rules.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        modified_memory = memory
        
        for name, rule in sorted_rules:
            # Check conditions
            if not all(cond(modified_memory) for cond in rule.conditions):
                continue
                
            # Apply rule based on type
            if rule.rule_type == RuleType.PATTERN:
                if rule.pattern.search(str(modified_memory.content)):
                    modified_memory = rule.action(modified_memory)
                    
            elif rule.rule_type == RuleType.TRANSFORMATION:
                cache_key = f"{name}_{hash(str(modified_memory.content))}"
                if cache_key in self.transformation_cache:
                    modified_memory = self.transformation_cache[cache_key]
                else:
                    transformed = rule.action(modified_memory)
                    self.transformation_cache[cache_key] = transformed
                    modified_memory = transformed
                    
            elif rule.rule_type == RuleType.INFERENCE:
                modified_memory = rule.action(modified_memory)
                
            elif rule.rule_type == RuleType.EMOTIONAL:
                if any(tag in modified_memory.emotional_tags for tag in [EmotionalTag.POSITIVE, EmotionalTag.NEGATIVE]):
                    modified_memory = rule.action(modified_memory)
                    
        return modified_memory
        
    def find_patterns(self, content: str) -> List[str]:
        """Find all matching patterns in content"""
        return self.pattern_matcher.match(content)
        
    def apply_transformations(self, memory: MemoryItem) -> MemoryItem:
        """Apply all transformation rules"""
        modified_memory = memory
        for rule in self.rules.values():
            if rule.rule_type == RuleType.TRANSFORMATION:
                modified_memory = rule.action(modified_memory)
        return modified_memory
        
    def make_inferences(self, memory: MemoryItem) -> List[MemoryItem]:
        """Generate inferences from memory"""
        inferences = []
        for rule in self.rules.values():
            if rule.rule_type == RuleType.INFERENCE:
                result = rule.action(memory)
                if isinstance(result, list):
                    inferences.extend(result)
                else:
                    inferences.append(result)
        return inferences

class RuleBuilder:
    """Helper class for building symbolic rules"""
    @staticmethod
    def create_pattern_rule(pattern: str,
                          action: Callable,
                          priority: int = 0) -> SymbolicRule:
        """Create pattern matching rule"""
        return SymbolicRule(
            rule_type=RuleType.PATTERN,
            pattern=re.compile(pattern),
            action=action,
            priority=priority,
            conditions=[],
            metadata={}
        )
        
    @staticmethod
    def create_transformation_rule(pattern: str,
                                 transform_func: Callable,
                                 priority: int = 0,
                                 conditions: List[Callable] = None) -> SymbolicRule:
        """Create transformation rule"""
        return SymbolicRule(
            rule_type=RuleType.TRANSFORMATION,
            pattern=re.compile(pattern),
            action=transform_func,
            priority=priority,
            conditions=conditions or [],
            metadata={}
        )
        
    @staticmethod
    def create_inference_rule(condition_func: Callable,
                            inference_func: Callable,
                            priority: int = 0) -> SymbolicRule:
        """Create inference rule"""
        return SymbolicRule(
            rule_type=RuleType.INFERENCE,
            pattern=re.compile(".*"),  # Match anything
            action=inference_func,
            priority=priority,
            conditions=[condition_func],
            metadata={}
        )
        
    @staticmethod
    def create_emotional_rule(emotion_pattern: str,
                            response_func: Callable,
                            priority: int = 0) -> SymbolicRule:
        """Create emotional response rule"""
        return SymbolicRule(
            rule_type=RuleType.EMOTIONAL,
            pattern=re.compile(emotion_pattern),
            action=response_func,
            priority=priority,
            conditions=[],
            metadata={}
        ) 