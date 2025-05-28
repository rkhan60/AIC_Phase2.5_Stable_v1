from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class MemoryQuery:
    """Structure for memory queries"""
    query_text: str
    context: Dict[str, Any]
    min_confidence: float = 0.5
    max_results: int = 10
    include_abstract: bool = True

@dataclass
class RetrievalResult:
    """Structure for memory retrieval results"""
    memory_id: str
    content: Dict[str, Any]
    confidence: float
    relevance_score: float
    memory_type: str  # 'concrete', 'abstract', 'consolidated'
    source_memories: Optional[List[str]] = None

class MemoryOptimizer:
    """Handles memory optimization and efficient retrieval"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.memory_embeddings: Dict[str, np.ndarray] = {}
        self.access_patterns: Dict[str, List[Tuple[datetime, float]]] = {}
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        
    def optimize_memory_layout(self, memory_manager) -> Dict[str, Any]:
        """Optimize memory storage layout based on access patterns"""
        # Analyze access patterns
        hot_memories = self._identify_hot_memories()
        cold_memories = self._identify_cold_memories()
        
        # Reorganize memory storage
        optimization_stats = {
            'hot_memories': len(hot_memories),
            'cold_memories': len(cold_memories),
            'cache_size': len(self.context_cache)
        }
        
        # Update cache with hot memories
        self._update_cache(hot_memories, memory_manager)
        
        # Pre-compute embeddings for frequently accessed memories
        self._precompute_embeddings(hot_memories, memory_manager)
        
        return optimization_stats
        
    def retrieve_relevant_memories(self, 
                                 query: MemoryQuery,
                                 memory_manager) -> List[RetrievalResult]:
        """Retrieve relevant memories using optimized search"""
        results = []
        
        # Generate query embedding
        query_embedding = self._get_text_embedding(query.query_text)
        
        # Search in concrete memories
        concrete_results = self._search_concrete_memories(
            query_embedding,
            query.context,
            memory_manager
        )
        results.extend(concrete_results)
        
        # Search in abstract patterns if requested
        if query.include_abstract:
            abstract_results = self._search_abstract_patterns(
                query_embedding,
                query.context,
                memory_manager
            )
            results.extend(abstract_results)
        
        # Sort by relevance and confidence
        results.sort(key=lambda x: (x.relevance_score * x.confidence), reverse=True)
        
        # Update access patterns
        self._update_access_patterns(results)
        
        return results[:query.max_results]
        
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using cached model"""
        return self.embedding_model.encode([text])[0]
        
    def _identify_hot_memories(self) -> List[str]:
        """Identify frequently accessed memories"""
        hot_memories = []
        now = datetime.now()
        
        for memory_id, accesses in self.access_patterns.items():
            # Filter recent accesses (last 24 hours)
            recent = [a for a in accesses if (now - a[0]).total_seconds() < 86400]
            
            # Calculate access frequency and average confidence
            if len(recent) > 5:  # More than 5 accesses in 24 hours
                avg_confidence = sum(a[1] for a in recent) / len(recent)
                if avg_confidence > 0.7:  # High confidence threshold
                    hot_memories.append(memory_id)
                    
        return hot_memories
        
    def _identify_cold_memories(self) -> List[str]:
        """Identify rarely accessed memories"""
        cold_memories = []
        now = datetime.now()
        
        for memory_id, accesses in self.access_patterns.items():
            # Check if no recent accesses (last 7 days)
            recent = [a for a in accesses if (now - a[0]).total_seconds() < 604800]
            if not recent:
                cold_memories.append(memory_id)
                
        return cold_memories
        
    def _update_cache(self, hot_memories: List[str], memory_manager):
        """Update context cache with hot memories"""
        # Clear old cache entries
        self.context_cache.clear()
        
        # Cache hot memories
        for memory_id in hot_memories:
            if memory_id in memory_manager.long_term_memory.items:
                item = memory_manager.long_term_memory.items[memory_id]
                self.context_cache[memory_id] = {
                    'content': item.content,
                    'metadata': item.metadata,
                    'confidence': item.confidence
                }
                
    def _precompute_embeddings(self, memory_ids: List[str], memory_manager):
        """Pre-compute embeddings for specified memories"""
        for memory_id in memory_ids:
            if memory_id in memory_manager.long_term_memory.items:
                item = memory_manager.long_term_memory.items[memory_id]
                # Create text representation of memory
                memory_text = self._memory_to_text(item.content)
                # Compute and store embedding
                self.memory_embeddings[memory_id] = self._get_text_embedding(memory_text)
                
    def _memory_to_text(self, memory_content: Dict[str, Any]) -> str:
        """Convert memory content to text representation"""
        text_parts = []
        
        # Add conditions
        if 'conditions' in memory_content:
            text_parts.append("Conditions: " + str(memory_content['conditions']))
            
        # Add consequences
        if 'consequences' in memory_content:
            text_parts.append("Consequences: " + str(memory_content['consequences']))
            
        # Add any additional metadata
        if 'metadata' in memory_content:
            text_parts.append("Context: " + str(memory_content['metadata']))
            
        return " ".join(text_parts)
        
    def _search_concrete_memories(self,
                                query_embedding: np.ndarray,
                                context: Dict[str, Any],
                                memory_manager) -> List[RetrievalResult]:
        """Search concrete memories using embeddings and context"""
        results = []
        
        for memory_id, item in memory_manager.long_term_memory.items.items():
            # Skip if confidence too low
            if item.confidence < 0.3:  # Minimum threshold
                continue
                
            # Get or compute memory embedding
            if memory_id not in self.memory_embeddings:
                memory_text = self._memory_to_text(item.content)
                memory_embedding = self._get_text_embedding(memory_text)
                self.memory_embeddings[memory_id] = memory_embedding
            else:
                memory_embedding = self.memory_embeddings[memory_id]
                
            # Calculate similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                memory_embedding.reshape(1, -1)
            )[0][0]
            
            # Check context match
            context_score = self._calculate_context_match(item.metadata, context)
            
            # Combined relevance score
            relevance = (similarity * 0.7) + (context_score * 0.3)
            
            if relevance > 0.5:  # Minimum relevance threshold
                results.append(RetrievalResult(
                    memory_id=memory_id,
                    content=item.content,
                    confidence=item.confidence,
                    relevance_score=relevance,
                    memory_type='concrete'
                ))
                
        return results
        
    def _search_abstract_patterns(self,
                                query_embedding: np.ndarray,
                                context: Dict[str, Any],
                                memory_manager) -> List[RetrievalResult]:
        """Search abstract patterns and rules"""
        results = []
        
        # Get abstract rules from rule engine
        abstract_rules = memory_manager.rule_engine.rules
        
        for rule in abstract_rules:
            if rule.project_context.get('type') == 'abstract_pattern':
                # Convert rule to text
                rule_text = f"Conditions: {rule.conditions} Consequences: {rule.consequences}"
                rule_embedding = self._get_text_embedding(rule_text)
                
                # Calculate similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    rule_embedding.reshape(1, -1)
                )[0][0]
                
                # Check context match
                context_score = self._calculate_context_match(
                    rule.project_context,
                    context
                )
                
                # Combined relevance score
                relevance = (similarity * 0.6) + (context_score * 0.4)
                
                if relevance > 0.4:  # Lower threshold for abstract patterns
                    results.append(RetrievalResult(
                        memory_id=f"abstract_rule_{id(rule)}",
                        content={
                            'conditions': rule.conditions,
                            'consequences': rule.consequences
                        },
                        confidence=rule.confidence,
                        relevance_score=relevance,
                        memory_type='abstract'
                    ))
                    
        return results
        
    def _calculate_context_match(self,
                               memory_context: Dict[str, Any],
                               query_context: Dict[str, Any]) -> float:
        """Calculate how well memory context matches query context"""
        if not memory_context or not query_context:
            return 0.5  # Neutral score if no context
            
        matching_keys = 0
        total_keys = len(query_context)
        
        for key, value in query_context.items():
            if key in memory_context:
                if memory_context[key] == value:
                    matching_keys += 1
                elif isinstance(value, (int, float)) and isinstance(memory_context[key], (int, float)):
                    # Allow for numeric ranges
                    diff = abs(memory_context[key] - value)
                    if diff / max(abs(value), 1) < 0.2:  # Within 20% range
                        matching_keys += 0.8
                        
        return matching_keys / total_keys
        
    def _update_access_patterns(self, results: List[RetrievalResult]):
        """Update access patterns for retrieved memories"""
        now = datetime.now()
        
        for result in results:
            if result.memory_id not in self.access_patterns:
                self.access_patterns[result.memory_id] = []
                
            self.access_patterns[result.memory_id].append(
                (now, result.confidence * result.relevance_score)
            )
            
            # Trim old access patterns (keep last 30 days)
            self.access_patterns[result.memory_id] = [
                a for a in self.access_patterns[result.memory_id]
                if (now - a[0]).total_seconds() < 2592000
            ] 