from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

@dataclass
class ContextualTrigger:
    """Represents a trigger for memory retrieval"""
    trigger_type: str  # 'structured' or 'semantic'
    content: Union[Dict[str, Any], str]
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class ContextTriggerSystem:
    """System for context-based memory retrieval"""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 semantic_threshold: float = 0.7):
        self.triggers: List[ContextualTrigger] = []
        self.semantic_model = SentenceTransformer(embedding_model)
        self.semantic_threshold = semantic_threshold
        
    def add_structured_trigger(self, 
                             trigger_content: Dict[str, Any],
                             metadata: Dict[str, Any] = None) -> ContextualTrigger:
        """Add a structured trigger (tags, tools, etc.)"""
        trigger = ContextualTrigger(
            trigger_type='structured',
            content=trigger_content,
            metadata=metadata or {},
            embedding=None
        )
        self.triggers.append(trigger)
        return trigger
        
    def add_semantic_trigger(self,
                           trigger_text: str,
                           metadata: Dict[str, Any] = None) -> ContextualTrigger:
        """Add a semantic trigger (text similarity)"""
        # Generate embedding
        embedding = self.semantic_model.encode(trigger_text)
        
        trigger = ContextualTrigger(
            trigger_type='semantic',
            content=trigger_text,
            metadata=metadata or {},
            embedding=embedding
        )
        self.triggers.append(trigger)
        return trigger
        
    def find_matching_triggers(self,
                             context: Union[Dict[str, Any], str],
                             trigger_type: Optional[str] = None) -> List[ContextualTrigger]:
        """Find triggers that match the given context"""
        matching_triggers = []
        
        # Filter by trigger type if specified
        triggers = self.triggers
        if trigger_type:
            triggers = [t for t in triggers if t.trigger_type == trigger_type]
            
        for trigger in triggers:
            if trigger.trigger_type == 'structured' and isinstance(context, dict):
                if self._match_structured_trigger(trigger, context):
                    matching_triggers.append(trigger)
                    
            elif trigger.trigger_type == 'semantic' and isinstance(context, str):
                if self._match_semantic_trigger(trigger, context):
                    matching_triggers.append(trigger)
                    
        return matching_triggers
        
    def _match_structured_trigger(self, trigger: ContextualTrigger, context: Dict[str, Any]) -> bool:
        """Match structured trigger against context"""
        trigger_content = trigger.content
        
        # Check each key-value pair in trigger
        for key, value in trigger_content.items():
            if key not in context:
                return False
                
            if isinstance(value, (str, int, float, bool)):
                if context[key] != value:
                    return False
            elif isinstance(value, list):
                if not any(v in context[key] for v in value):
                    return False
            elif isinstance(value, dict):
                if not self._match_structured_trigger(
                    ContextualTrigger('structured', value, {}),
                    context[key]
                ):
                    return False
                    
        return True
        
    def _match_semantic_trigger(self, trigger: ContextualTrigger, context: str) -> bool:
        """Match semantic trigger against context"""
        # Generate embedding for context
        context_embedding = self.semantic_model.encode(context)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            torch.tensor(trigger.embedding).unsqueeze(0),
            torch.tensor(context_embedding).unsqueeze(0)
        ).item()
        
        return similarity >= self.semantic_threshold
        
    def get_best_semantic_matches(self,
                                context: str,
                                top_k: int = 5) -> List[tuple[ContextualTrigger, float]]:
        """Get top-k semantic matches sorted by similarity"""
        semantic_triggers = [t for t in self.triggers if t.trigger_type == 'semantic']
        if not semantic_triggers:
            return []
            
        # Generate embedding for context
        context_embedding = self.semantic_model.encode(context)
        
        # Calculate similarities
        similarities = []
        for trigger in semantic_triggers:
            similarity = F.cosine_similarity(
                torch.tensor(trigger.embedding).unsqueeze(0),
                torch.tensor(context_embedding).unsqueeze(0)
            ).item()
            similarities.append((trigger, similarity))
            
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k] 