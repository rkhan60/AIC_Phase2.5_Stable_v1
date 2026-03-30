from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
import json
from pathlib import Path
import threading
import queue

class MemoryType(Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EMOTIONAL = "emotional"

class EmotionalTag(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    URGENT = "urgent"
    IMPORTANT = "important"

@dataclass
class MemoryItem:
    """Base class for memory items"""
    content: Any
    memory_type: MemoryType
    context_tags: List[str]
    emotional_tags: List[EmotionalTag]
    importance_score: float
    last_accessed: Optional[datetime]
    access_count: int = 1
    created_at: datetime = datetime.now()
    related_memories: List[str] = None  # IDs of related memories
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.related_memories is None:
            self.related_memories = []
        if self.metadata is None:
            self.metadata = {}

    def update_access_time(self):
        """Update the last accessed time"""
        self.last_accessed = datetime.now()
        
    def increase_importance(self, factor: float = 0.1):
        """Increase the importance score"""
        self.importance_score = min(1.0, self.importance_score + factor)
        
    def decrease_importance(self, factor: float = 0.1):
        """Decrease the importance score"""
        self.importance_score = max(0.0, self.importance_score - factor)

class MemoryBuffer:
    """Buffer for temporary memory storage"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: List[MemoryItem] = []
        self.lock = threading.Lock()

    def add(self, item: MemoryItem) -> bool:
        with self.lock:
            if len(self.items) >= self.capacity:
                self._consolidate()
            self.items.append(item)
            return True

    def _consolidate(self):
        """Consolidate buffer by importance and recency"""
        self.items.sort(key=lambda x: (x.importance_score, x.last_accessed), reverse=True)
        self.items = self.items[:self.capacity - 1]

class MemoryStore:
    """Persistent storage for memories"""
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memory_index: Dict[str, MemoryItem] = {}
        self.lock = threading.Lock()

    def save(self, item: MemoryItem) -> str:
        """Save memory item to persistent storage"""
        memory_id = self._generate_id(item)
        with self.lock:
            self.memory_index[memory_id] = item
            self._persist_to_disk(memory_id, item)
        return memory_id

    def _generate_id(self, item: MemoryItem) -> str:
        """Generate unique ID for memory item"""
        import hashlib
        content_hash = hashlib.md5(str(item.content).encode()).hexdigest()
        timestamp = item.created_at.strftime("%Y%m%d%H%M%S")
        return f"{item.memory_type.value}_{timestamp}_{content_hash[:8]}"

    def _persist_to_disk(self, memory_id: str, item: MemoryItem):
        """Save memory to disk"""
        file_path = self.storage_path / f"{memory_id}.json"
        with open(file_path, 'w') as f:
            json.dump({
                'content': item.content,
                'created_at': item.created_at.isoformat(),
                'memory_type': item.memory_type.value,
                'last_accessed': item.last_accessed.isoformat(),
                'access_count': item.access_count,
                'emotional_tags': [tag.value for tag in item.emotional_tags],
                'importance_score': item.importance_score,
                'context_tags': item.context_tags,
                'related_memories': item.related_memories,
                'metadata': item.metadata
            }, f)

class MemorySystem:
    """Core memory management system"""
    def __init__(self,
                 working_memory_capacity: int = 100,
                 storage_path: Path = Path("./memory_store")):
        self.working_memory = MemoryBuffer(working_memory_capacity)
        self.long_term_store = MemoryStore(storage_path)
        self.emotional_processor = EmotionalProcessor()
        self.consolidation_queue = queue.PriorityQueue()
        self._stop_consolidation = threading.Event()
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background processing threads"""
        self.consolidation_thread = threading.Thread(
            target=self._consolidation_worker,
            daemon=True,
            name="MemoryConsolidationWorker",
        )
        self.consolidation_thread.start()

    def _consolidation_worker(self):
        """Background worker for memory consolidation"""
        while not self._stop_consolidation.is_set():
            try:
                # Use timeout so the loop can check the stop event regularly
                try:
                    priority, memory = self.consolidation_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                self._process_consolidation(memory)
                self.consolidation_queue.task_done()
            except Exception as exc:
                import logging as _logging
                _logging.getLogger(__name__).error(
                    "Error in consolidation worker: %s", exc
                )

    def shutdown(self):
        """Gracefully stop the background consolidation thread."""
        self._stop_consolidation.set()
        self.consolidation_thread.join(timeout=5.0)

    def _process_consolidation(self, memory: MemoryItem):
        """Process memory consolidation"""
        # Update importance based on emotional tags and access patterns
        emotional_importance = self.emotional_processor.calculate_importance(memory)
        access_importance = min(1.0, memory.access_count / 100)
        time_decay = self._calculate_time_decay(memory)

        memory.importance_score = (
            emotional_importance * 0.4 +
            access_importance * 0.3 +
            time_decay * 0.3
        )

        # Store in long-term memory if important enough
        if memory.importance_score > 0.5:
            self.long_term_store.save(memory)

    def _calculate_time_decay(self, memory: MemoryItem) -> float:
        """Calculate time-based decay factor"""
        age = datetime.now() - memory.created_at
        # Exponential decay with half-life of 30 days
        half_life = timedelta(days=30)
        decay = 2 ** (-age / half_life)
        return max(0.1, decay)

    def add_memory(self, 
                  content: Any,
                  memory_type: MemoryType,
                  emotional_tags: List[EmotionalTag] = None,
                  context_tags: List[str] = None,
                  metadata: Dict[str, Any] = None) -> str:
        """Add new memory to the system"""
        now = datetime.now()
        memory = MemoryItem(
            content=content,
            created_at=now,
            memory_type=memory_type,
            last_accessed=now,
            access_count=1,
            emotional_tags=emotional_tags or [EmotionalTag.NEUTRAL],
            importance_score=0.5,  # Initial score
            context_tags=context_tags or [],
            related_memories=[],
            metadata=metadata or {}
        )

        # Add to working memory
        if memory_type == MemoryType.WORKING:
            self.working_memory.add(memory)
        
        # Queue for consolidation
        self.consolidation_queue.put((
            -memory.importance_score,  # Negative for priority queue
            memory
        ))

        # Store immediately if emotional or long-term
        if memory_type in [MemoryType.EMOTIONAL, MemoryType.LONG_TERM]:
            return self.long_term_store.save(memory)

        return ""

    def store_episode(self, episode: Dict[str, Any]) -> bool:
        """Store a complete reasoning-cycle episode in long-term memory.

        Args:
            episode: Dict containing cycle inputs, outputs, and metadata.

        Returns:
            True on success, False on failure.
        """
        try:
            validation_status = str(episode.get("validation", "unknown"))
            self.add_memory(
                content=episode,
                memory_type=MemoryType.LONG_TERM,
                emotional_tags=[EmotionalTag.IMPORTANT],
                context_tags=["episode", validation_status],
                metadata={"stored_at": datetime.now().isoformat()},
            )
            return True
        except Exception as exc:
            print(f"Failed to store episode: {exc}")
            return False

    def retrieve_memory(self,
                       memory_id: str = None,
                       context_tags: List[str] = None,
                       emotional_tags: List[EmotionalTag] = None) -> List[MemoryItem]:
        """Retrieve memories based on criteria"""
        results = []

        # Direct lookup by ID
        if memory_id and memory_id in self.long_term_store.memory_index:
            memory = self.long_term_store.memory_index[memory_id]
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            results.append(memory)
            return results

        # Search by tags
        for memory in self.long_term_store.memory_index.values():
            matches_context = not context_tags or any(tag in memory.context_tags for tag in context_tags)
            matches_emotion = not emotional_tags or any(tag in memory.emotional_tags for tag in emotional_tags)

            if matches_context or matches_emotion:
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                results.append(memory)

        return results

class EmotionalProcessor:
    """Process and analyze emotional aspects of memories"""
    def calculate_importance(self, memory: MemoryItem) -> float:
        """Calculate emotional importance score"""
        emotional_weights = {
            EmotionalTag.URGENT: 1.0,
            EmotionalTag.IMPORTANT: 0.8,
            EmotionalTag.POSITIVE: 0.6,
            EmotionalTag.NEGATIVE: 0.7,
            EmotionalTag.NEUTRAL: 0.3
        }

        # Calculate average emotional weight
        total_weight = sum(emotional_weights[tag] for tag in memory.emotional_tags)
        return total_weight / len(memory.emotional_tags) 