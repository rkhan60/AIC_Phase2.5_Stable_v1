from typing import List, Dict, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from .memory_system import MemoryItem, MemoryType
from .context_system import ContextualTriggerSystem

logger = logging.getLogger(__name__)

class MemoryQueryEngine:
    """Engine for querying memory based on semantic and contextual patterns"""
    def __init__(self, trigger_system: Optional[ContextualTriggerSystem] = None):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.trigger_system = trigger_system
        self.query_history: List[Dict[str, Any]] = []
        self.semantic_cache: Dict[str, np.ndarray] = {}
        
    def find_similar_experiences(self,
                               query: str,
                               memories: List[MemoryItem],
                               threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find memories similar to the query"""
        query_embedding = self._get_embedding(query)
        similar_memories = []
        
        for memory in memories:
            similarity = self._calculate_similarity(query_embedding, memory)
            if similarity >= threshold:
                similar_memories.append({
                    'memory': memory,
                    'similarity': similarity,
                    'timestamp': datetime.now()
                })
                
        # Sort by similarity
        similar_memories.sort(key=lambda x: x['similarity'], reverse=True)
        
        self._log_query('similar_experiences', query, similar_memories)
        return similar_memories
        
    def find_failure_patterns(self,
                            input_data: Any,
                            memories: List[MemoryItem]) -> List[Dict[str, Any]]:
        """Find patterns in past failures similar to input"""
        # Convert input to string for comparison
        input_str = str(input_data)
        input_embedding = self._get_embedding(input_str)
        
        # Filter for failure-related memories
        failure_memories = [
            m for m in memories
            if any(tag.value == 'negative' for tag in m.emotional_tags)
        ]
        
        patterns = []
        for memory in failure_memories:
            similarity = self._calculate_similarity(input_embedding, memory)
            if similarity >= 0.6:  # Lower threshold for failure detection
                patterns.append({
                    'memory': memory,
                    'similarity': similarity,
                    'warning': self._generate_warning(memory),
                    'timestamp': datetime.now()
                })
                
        self._log_query('failure_patterns', input_str, patterns)
        return patterns
        
    def query_successful_strategies(self,
                                  context: str,
                                  memories: List[MemoryItem]) -> List[Dict[str, Any]]:
        """Find successful strategies in similar contexts"""
        context_embedding = self._get_embedding(context)
        
        # Filter for success-related memories
        success_memories = [
            m for m in memories
            if any(tag.value == 'positive' for tag in m.emotional_tags)
        ]
        
        strategies = []
        for memory in success_memories:
            similarity = self._calculate_similarity(context_embedding, memory)
            if similarity >= 0.7:
                strategies.append({
                    'memory': memory,
                    'similarity': similarity,
                    'strategy': self._extract_strategy(memory),
                    'timestamp': datetime.now()
                })
                
        self._log_query('successful_strategies', context, strategies)
        return strategies
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get or compute embedding for text"""
        if text in self.semantic_cache:
            return self.semantic_cache[text]
            
        embedding = self.model.encode([text])[0]
        self.semantic_cache[text] = embedding
        return embedding
        
    def _calculate_similarity(self, query_embedding: np.ndarray, memory: MemoryItem) -> float:
        """Calculate semantic similarity between query and memory"""
        memory_embedding = self._get_embedding(str(memory.content))
        similarity = np.dot(query_embedding, memory_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
        return float(similarity)
        
    def _generate_warning(self, memory: MemoryItem) -> str:
        """Generate warning message from failure memory"""
        return f"Similar failure detected: {memory.content} (Importance: {memory.importance_score})"
        
    def _extract_strategy(self, memory: MemoryItem) -> str:
        """Extract successful strategy from memory"""
        return f"Successful approach: {memory.content} (Used {memory.access_count} times)"
        
    def _log_query(self, query_type: str, query: str, results: List[Dict[str, Any]]):
        """Log query and results"""
        self.query_history.append({
            'timestamp': datetime.now(),
            'query_type': query_type,
            'query': query,
            'num_results': len(results),
            'avg_similarity': np.mean([r['similarity'] for r in results]) if results else 0.0
        })
        
    def get_query_stats(self) -> Dict[str, Any]:
        """Get statistics about memory queries"""
        if not self.query_history:
            return {}
            
        stats = {
            'total_queries': len(self.query_history),
            'query_types': {},
            'avg_results': 0,
            'avg_similarity': 0.0,
            'last_query': self.query_history[-1]['timestamp']
        }
        
        # Calculate query type distribution
        for query in self.query_history:
            query_type = query['query_type']
            stats['query_types'][query_type] = stats['query_types'].get(query_type, 0) + 1
            
        # Calculate averages
        stats['avg_results'] = np.mean([q['num_results'] for q in self.query_history])
        stats['avg_similarity'] = np.mean([q['avg_similarity'] for q in self.query_history])
        
        return stats 