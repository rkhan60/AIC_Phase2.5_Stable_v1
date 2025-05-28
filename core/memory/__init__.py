from pathlib import Path
from typing import Optional

from .symbolic_rule_engine import SymbolicRule, SymbolicRuleEngine
from .memory_manager import MemoryManager, MemoryItem, WorkingMemory, LongTermMemory, EmotionalMemory
from .context_trigger import ContextualTrigger, ContextTriggerSystem

def initialize_memory_system(base_path: Optional[Path] = None) -> Path:
    """Initialize the memory system directory structure"""
    base_path = base_path or Path("memory_store")
    
    # Create main directories
    directories = [
        base_path,
        base_path / "working_memory",
        base_path / "long_term_memory",
        base_path / "emotional_memory",
        base_path / "rules",
        base_path / "triggers"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    return base_path

__all__ = [
    'SymbolicRule',
    'SymbolicRuleEngine',
    'MemoryManager',
    'MemoryItem',
    'WorkingMemory',
    'LongTermMemory',
    'EmotionalMemory',
    'ContextualTrigger',
    'ContextTriggerSystem',
    'initialize_memory_system'
]

"""
Memory system package containing core memory management functionality
""" 