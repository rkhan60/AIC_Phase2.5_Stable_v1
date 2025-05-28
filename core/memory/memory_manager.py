from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from .symbolic_rule_engine import SymbolicRuleEngine
from .decay_manager import DecayManager, DecayConfig

@dataclass
class MemoryItem:
    """Base class for memory items"""
    content: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    metadata: Dict[str, Any]
    confidence: float = 1.0  # Added confidence score

@dataclass
class WorkingMemory:
    """Short-term task-specific memory"""
    items: Dict[str, MemoryItem]
    max_size: int = 100
    
    def add_item(self, key: str, content: Any, metadata: Dict[str, Any] = None):
        """Add item to working memory"""
        if len(self.items) >= self.max_size:
            # Remove oldest item
            oldest_key = min(self.items.keys(), key=lambda k: self.items[k].last_accessed)
            del self.items[oldest_key]
            
        self.items[key] = MemoryItem(
            content=content,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            metadata=metadata or {}
        )
        
    def get_item(self, key: str) -> Optional[Any]:
        """Retrieve item from working memory"""
        if key in self.items:
            item = self.items[key]
            item.last_accessed = datetime.now()
            item.access_count += 1
            return item.content
        return None

@dataclass
class LongTermMemory:
    """Strategic patterns and insights across projects"""
    items: Dict[str, MemoryItem]
    rule_engine: SymbolicRuleEngine
    
    def add_pattern(self, pattern_key: str, pattern: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Add strategic pattern to long-term memory"""
        self.items[pattern_key] = MemoryItem(
            content=pattern,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            metadata=metadata or {}
        )
        
        # Create symbolic rule if pattern contains conditions and consequences
        if 'conditions' in pattern and 'consequences' in pattern:
            self.rule_engine.add_rule(
                conditions=pattern['conditions'],
                consequences=pattern['consequences'],
                project_context=metadata or {},
                confidence=pattern.get('confidence', 1.0)
            )
            
    def get_pattern(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve strategic pattern"""
        if pattern_key in self.items:
            item = self.items[pattern_key]
            item.last_accessed = datetime.now()
            item.access_count += 1
            return item.content
        return None
        
    def find_relevant_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find patterns relevant to given context"""
        matching_rules = self.rule_engine.find_matching_rules(context)
        relevant_patterns = []
        
        for rule in matching_rules:
            pattern = {
                'conditions': rule.conditions,
                'consequences': rule.consequences,
                'confidence': rule.confidence,
                'use_count': rule.use_count
            }
            relevant_patterns.append(pattern)
            
        return relevant_patterns

@dataclass
class EmotionalMemory:
    """Qualitative feedback and satisfaction indicators"""
    items: Dict[str, MemoryItem]
    
    def add_feedback(self, 
                    feedback_key: str, 
                    feedback: Dict[str, Any],
                    sentiment: float,  # -1.0 to 1.0
                    metadata: Dict[str, Any] = None):
        """Add emotional feedback to memory"""
        self.items[feedback_key] = MemoryItem(
            content={
                'feedback': feedback,
                'sentiment': sentiment
            },
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            metadata=metadata or {}
        )
        
    def get_feedback(self, feedback_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve emotional feedback"""
        if feedback_key in self.items:
            item = self.items[feedback_key]
            item.last_accessed = datetime.now()
            item.access_count += 1
            return item.content
        return None
        
    def get_sentiment_summary(self, context_filter: Dict[str, Any] = None) -> Dict[str, float]:
        """Get sentiment summary, optionally filtered by context"""
        relevant_items = self.items
        if context_filter:
            relevant_items = {
                k: v for k, v in self.items.items()
                if all(v.metadata.get(key) == value for key, value in context_filter.items())
            }
            
        if not relevant_items:
            return {'average_sentiment': 0.0, 'feedback_count': 0}
            
        sentiments = [item.content['sentiment'] for item in relevant_items.values()]
        return {
            'average_sentiment': sum(sentiments) / len(sentiments),
            'feedback_count': len(sentiments)
        }

class MemoryManager:
    """Central manager for all memory types"""
    
    def __init__(self, storage_path: Optional[Path] = None, decay_config: Optional[DecayConfig] = None):
        self.storage_path = storage_path or Path("memory_store")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.rule_engine = SymbolicRuleEngine(self.storage_path / "rules.json")
        self.decay_manager = DecayManager(decay_config)
        
        self.working_memory = WorkingMemory(items={})
        self.long_term_memory = LongTermMemory(items={}, rule_engine=self.rule_engine)
        self.emotional_memory = EmotionalMemory(items={})
        
    def apply_decay_cycle(self):
        """Apply decay to all memory items"""
        if not self.decay_manager.should_check_decay():
            return
            
        # Get memory statistics
        stats = self._calculate_memory_stats()
        
        # Update decay parameters based on system state
        adaptive_params = self.decay_manager.get_adaptive_parameters(stats)
        self.decay_manager.update_decay_config(adaptive_params)
        
        # Apply decay to long-term memory
        self._decay_long_term_memory()
        
        # Apply decay to emotional memory
        self._decay_emotional_memory()
        
        # Mark decay check complete
        self.decay_manager.mark_decay_check()
        
    def _decay_long_term_memory(self):
        """Apply decay to long-term memory items"""
        items_to_remove = []
        
        for key, item in self.long_term_memory.items.items():
            # Calculate new confidence
            new_confidence = self.decay_manager.apply_decay(
                item.confidence,
                item.created_at,
                item.last_accessed,
                item.access_count
            )
            
            # Check if item should be pruned
            if self.decay_manager.should_prune(
                new_confidence,
                item.last_accessed,
                item.access_count
            ):
                items_to_remove.append(key)
            else:
                item.confidence = new_confidence
                
        # Remove pruned items
        for key in items_to_remove:
            del self.long_term_memory.items[key]
            
    def _decay_emotional_memory(self):
        """Apply decay to emotional memory items"""
        items_to_remove = []
        
        for key, item in self.emotional_memory.items.items():
            # Calculate new confidence
            new_confidence = self.decay_manager.apply_decay(
                item.confidence,
                item.created_at,
                item.last_accessed,
                item.access_count
            )
            
            # Check if item should be pruned
            if self.decay_manager.should_prune(
                new_confidence,
                item.last_accessed,
                item.access_count
            ):
                items_to_remove.append(key)
            else:
                item.confidence = new_confidence
                
        # Remove pruned items
        for key in items_to_remove:
            del self.emotional_memory.items[key]
            
    def _calculate_memory_stats(self) -> Dict[str, Any]:
        """Calculate memory system statistics"""
        long_term_items = list(self.long_term_memory.items.values())
        emotional_items = list(self.emotional_memory.items.values())
        all_items = long_term_items + emotional_items
        
        if not all_items:
            return {
                'avg_confidence': 0.5,
                'total_memories': 0,
                'avg_age': 0
            }
            
        # Calculate statistics
        total_items = len(all_items)
        avg_confidence = sum(item.confidence for item in all_items) / total_items
        
        # Calculate average age in days
        total_age = sum(
            (datetime.now() - item.created_at).total_seconds() / (24 * 3600)
            for item in all_items
        )
        avg_age = total_age / total_items
        
        return {
            'avg_confidence': avg_confidence,
            'total_memories': total_items,
            'avg_age': avg_age
        }
        
    def reinforce_memory(self, memory_key: str, success_score: float, memory_type: str = 'long_term'):
        """Reinforce a memory item based on successful use"""
        memory_dict = (
            self.long_term_memory.items if memory_type == 'long_term'
            else self.emotional_memory.items
        )
        
        if memory_key in memory_dict:
            item = memory_dict[memory_key]
            item.confidence = self.decay_manager.apply_reinforcement(
                item.confidence,
                success_score,
                item.access_count
            )
            item.last_accessed = datetime.now()
            item.access_count += 1
            
    def get_memory_weight(self, memory_key: str, memory_type: str = 'long_term') -> float:
        """Get the importance weight of a memory item"""
        memory_dict = (
            self.long_term_memory.items if memory_type == 'long_term'
            else self.emotional_memory.items
        )
        
        if memory_key in memory_dict:
            item = memory_dict[memory_key]
            return self.decay_manager.calculate_memory_weight(
                item.confidence,
                item.created_at,
                item.last_accessed,
                item.access_count
            )
        return 0.0
        
    def save_state(self):
        """Save all memory states"""
        # Apply decay cycle before saving
        self.apply_decay_cycle()
        
        # Save rules
        self.rule_engine.save_rules()
        
        # Save memory items
        memory_data = {
            'working_memory': self._serialize_memory_items(self.working_memory.items),
            'long_term_memory': self._serialize_memory_items(self.long_term_memory.items),
            'emotional_memory': self._serialize_memory_items(self.emotional_memory.items)
        }
        
        with open(self.storage_path / "memory_state.json", 'w') as f:
            json.dump(memory_data, f, indent=2)
            
    def load_state(self):
        """Load all memory states"""
        # Load rules
        self.rule_engine.load_rules()
        
        # Load memory items
        if (self.storage_path / "memory_state.json").exists():
            with open(self.storage_path / "memory_state.json") as f:
                memory_data = json.load(f)
                
            self.working_memory.items = self._deserialize_memory_items(memory_data['working_memory'])
            self.long_term_memory.items = self._deserialize_memory_items(memory_data['long_term_memory'])
            self.emotional_memory.items = self._deserialize_memory_items(memory_data['emotional_memory'])
            
    def _serialize_memory_items(self, items: Dict[str, MemoryItem]) -> Dict[str, Dict]:
        """Serialize memory items for storage"""
        serialized = {}
        for key, item in items.items():
            serialized[key] = {
                'content': item.content,
                'created_at': item.created_at.isoformat(),
                'last_accessed': item.last_accessed.isoformat(),
                'access_count': item.access_count,
                'metadata': item.metadata,
                'confidence': item.confidence
            }
        return serialized
        
    def _deserialize_memory_items(self, data: Dict[str, Dict]) -> Dict[str, MemoryItem]:
        """Deserialize memory items from storage"""
        items = {}
        for key, item_data in data.items():
            items[key] = MemoryItem(
                content=item_data['content'],
                created_at=datetime.fromisoformat(item_data['created_at']),
                last_accessed=datetime.fromisoformat(item_data['last_accessed']),
                access_count=item_data['access_count'],
                metadata=item_data['metadata'],
                confidence=item_data.get('confidence', 1.0)
            )
        return items 

    def consolidate_memories(self):
        """Smart memory consolidation that transforms instead of deleting"""
        # Get memory statistics
        stats = self._calculate_memory_stats()
        
        # Identify similar memories for consolidation
        consolidated = self._consolidate_similar_patterns()
        
        # Abstract common patterns into higher-level rules
        abstracted = self._abstract_patterns()
        
        # Transform rarely used but potentially valuable memories
        transformed = self._transform_rare_memories()
        
        return {
            'consolidated': consolidated,
            'abstracted': abstracted,
            'transformed': transformed
        }
        
    def _consolidate_similar_patterns(self) -> List[Dict[str, Any]]:
        """Combine similar memory patterns into stronger consolidated memories"""
        consolidated_patterns = []
        pattern_groups = {}
        
        # Group similar patterns
        for key, item in self.long_term_memory.items.items():
            pattern = item.content
            pattern_hash = self._calculate_pattern_similarity_hash(pattern)
            
            if pattern_hash not in pattern_groups:
                pattern_groups[pattern_hash] = []
            pattern_groups[pattern_hash].append((key, item))
            
        # Consolidate each group
        for pattern_hash, items in pattern_groups.items():
            if len(items) > 1:
                # Merge similar patterns
                merged_pattern = self._merge_patterns([item[1].content for item in items])
                merged_confidence = max(item[1].confidence for item in items)
                
                # Create consolidated memory
                consolidated_key = f"consolidated_{pattern_hash}"
                self.long_term_memory.add_pattern(
                    consolidated_key,
                    merged_pattern,
                    metadata={'source_patterns': [item[0] for item in items]}
                )
                
                # Remove original patterns
                for key, _ in items:
                    del self.long_term_memory.items[key]
                    
                consolidated_patterns.append(merged_pattern)
                
        return consolidated_patterns
        
    def _abstract_patterns(self) -> List[Dict[str, Any]]:
        """Abstract common patterns into higher-level rules"""
        abstracted_rules = []
        
        # Find common elements across patterns
        common_elements = self._find_common_pattern_elements()
        
        for element_key, occurrences in common_elements.items():
            if occurrences > 3:  # Abstract patterns that appear frequently
                # Create higher-level abstract rule
                abstract_rule = self._create_abstract_rule(element_key)
                
                if abstract_rule:
                    self.rule_engine.add_rule(
                        conditions=abstract_rule['conditions'],
                        consequences=abstract_rule['consequences'],
                        project_context={'type': 'abstract_pattern'},
                        confidence=0.8  # Start with reasonable confidence
                    )
                    abstracted_rules.append(abstract_rule)
                    
        return abstracted_rules
        
    def _transform_rare_memories(self) -> List[Dict[str, Any]]:
        """Transform rarely used but potentially valuable memories"""
        transformed = []
        
        for key, item in list(self.long_term_memory.items.items()):
            if item.access_count < 3 and item.confidence < 0.5:
                # Instead of deletion, transform into a more general pattern
                generalized = self._generalize_pattern(item.content)
                
                if generalized:
                    # Store transformed pattern
                    new_key = f"transformed_{key}"
                    self.long_term_memory.add_pattern(
                        new_key,
                        generalized,
                        metadata={
                            'transformed_from': key,
                            'transformation_type': 'generalization'
                        }
                    )
                    transformed.append(generalized)
                    
                    # Remove original specific pattern
                    del self.long_term_memory.items[key]
                    
        return transformed
        
    def _calculate_pattern_similarity_hash(self, pattern: Dict[str, Any]) -> str:
        """Calculate similarity hash for pattern matching"""
        # Implement similarity hashing based on key pattern elements
        # This is a simplified version - expand based on your specific pattern structure
        key_elements = []
        
        if 'conditions' in pattern:
            key_elements.extend(sorted(pattern['conditions'].keys()))
        if 'consequences' in pattern:
            key_elements.extend(sorted(pattern['consequences'].keys()))
            
        return '_'.join(key_elements)
        
    def _merge_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge similar patterns into a consolidated pattern"""
        # Implement pattern merging logic
        # This is a simplified version - expand based on your pattern structure
        merged = {
            'conditions': {},
            'consequences': {},
            'metadata': {'merged_count': len(patterns)}
        }
        
        for pattern in patterns:
            for key, value in pattern.get('conditions', {}).items():
                if key not in merged['conditions']:
                    merged['conditions'][key] = []
                if value not in merged['conditions'][key]:
                    merged['conditions'][key].append(value)
                    
            for key, value in pattern.get('consequences', {}).items():
                if key not in merged['consequences']:
                    merged['consequences'][key] = []
                if value not in merged['consequences'][key]:
                    merged['consequences'][key].append(value)
                    
        return merged
        
    def _find_common_pattern_elements(self) -> Dict[str, int]:
        """Find common elements across patterns"""
        element_counts = {}
        
        for item in self.long_term_memory.items.values():
            pattern = item.content
            elements = self._extract_pattern_elements(pattern)
            
            for element in elements:
                element_counts[element] = element_counts.get(element, 0) + 1
                
        return element_counts
        
    def _extract_pattern_elements(self, pattern: Dict[str, Any]) -> List[str]:
        """Extract key elements from a pattern"""
        elements = []
        
        def extract_recursive(obj: Any, prefix: str = ''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    elements.append(full_key)
                    extract_recursive(value, full_key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{prefix}[{i}]")
                    
        extract_recursive(pattern)
        return elements
        
    def _create_abstract_rule(self, element_key: str) -> Optional[Dict[str, Any]]:
        """Create an abstract rule from common pattern elements"""
        # Implement abstract rule creation logic
        # This is a simplified version - expand based on your needs
        related_patterns = []
        
        for item in self.long_term_memory.items.values():
            pattern = item.content
            if element_key in self._extract_pattern_elements(pattern):
                related_patterns.append(pattern)
                
        if not related_patterns:
            return None
            
        # Create abstract rule from common elements
        abstract_rule = {
            'conditions': {
                'abstract_type': element_key,
                'pattern_count': len(related_patterns)
            },
            'consequences': {
                'suggested_actions': self._extract_common_actions(related_patterns)
            }
        }
        
        return abstract_rule
        
    def _extract_common_actions(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Extract common actions from related patterns"""
        action_counts = {}
        
        for pattern in patterns:
            for action in pattern.get('consequences', {}).get('actions', []):
                action_counts[action] = action_counts.get(action, 0) + 1
                
        # Return actions that appear in majority of patterns
        threshold = len(patterns) / 2
        return [action for action, count in action_counts.items() if count > threshold]
        
    def _generalize_pattern(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generalize a specific pattern into a more abstract form"""
        if not pattern.get('conditions') or not pattern.get('consequences'):
            return None
            
        # Create more general conditions
        general_conditions = {}
        for key, value in pattern['conditions'].items():
            if isinstance(value, (int, float)):
                # Convert specific numbers to ranges
                general_conditions[key] = {
                    'range': [value * 0.8, value * 1.2]
                }
            elif isinstance(value, str):
                # Extract key terms or patterns
                general_conditions[key] = {
                    'pattern': self._extract_key_terms(value)
                }
            else:
                general_conditions[key] = value
                
        # Generalize consequences
        general_consequences = {
            'action_type': self._categorize_actions(pattern['consequences']),
            'expected_outcomes': self._generalize_outcomes(pattern['consequences'])
        }
        
        return {
            'conditions': general_conditions,
            'consequences': general_consequences,
            'metadata': {
                'generalized_from': pattern.get('metadata', {}),
                'generalization_type': 'range_based'
            }
        }
        
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Implement key term extraction
        # This is a simplified version - expand based on your needs
        return [term.strip() for term in text.split() if len(term.strip()) > 3]
        
    def _categorize_actions(self, consequences: Dict[str, Any]) -> str:
        """Categorize action types"""
        # Implement action categorization
        # This is a simplified version - expand based on your needs
        if 'update' in str(consequences).lower():
            return 'modification'
        elif 'create' in str(consequences).lower():
            return 'creation'
        elif 'delete' in str(consequences).lower():
            return 'removal'
        return 'general'
        
    def _generalize_outcomes(self, consequences: Dict[str, Any]) -> Dict[str, Any]:
        """Generalize expected outcomes"""
        # Implement outcome generalization
        # This is a simplified version - expand based on your needs
        return {
            'impact_area': self._identify_impact_area(consequences),
            'confidence_range': [0.6, 0.9]  # Conservative confidence range
        }
        
    def _identify_impact_area(self, consequences: Dict[str, Any]) -> str:
        """Identify the area impacted by consequences"""
        # Implement impact area identification
        # This is a simplified version - expand based on your needs
        impact_terms = {
            'data': ['database', 'storage', 'record'],
            'ui': ['interface', 'display', 'view'],
            'logic': ['process', 'calculate', 'compute'],
            'system': ['config', 'setting', 'parameter']
        }
        
        consequences_str = str(consequences).lower()
        for area, terms in impact_terms.items():
            if any(term in consequences_str for term in terms):
                return area
        return 'general' 