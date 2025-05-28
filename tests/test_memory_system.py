import unittest
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
from core.memory.memory_system import (
    MemorySystem,
    MemoryType,
    EmotionalTag,
    MemoryItem
)
from core.memory.symbolic_engine import (
    SymbolicEngine,
    RuleType,
    RuleBuilder
)

class TestMemorySystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory_system = MemorySystem(
            working_memory_capacity=10,
            storage_path=Path(self.temp_dir)
        )
        
    def test_memory_addition(self):
        """Test adding memories of different types"""
        # Add working memory
        working_id = self.memory_system.add_memory(
            content="Test working memory",
            memory_type=MemoryType.WORKING,
            emotional_tags=[EmotionalTag.NEUTRAL],
            context_tags=["test"]
        )
        
        # Add emotional memory
        emotional_id = self.memory_system.add_memory(
            content="Test emotional memory",
            memory_type=MemoryType.EMOTIONAL,
            emotional_tags=[EmotionalTag.POSITIVE],
            context_tags=["test", "emotional"]
        )
        
        # Add long-term memory
        longterm_id = self.memory_system.add_memory(
            content="Test long-term memory",
            memory_type=MemoryType.LONG_TERM,
            emotional_tags=[EmotionalTag.IMPORTANT],
            context_tags=["test", "long-term"]
        )
        
        # Verify storage
        self.assertEqual(len(self.memory_system.working_memory.items), 1)
        self.assertIn(emotional_id, self.memory_system.long_term_store.memory_index)
        self.assertIn(longterm_id, self.memory_system.long_term_store.memory_index)
        
    def test_memory_retrieval(self):
        """Test memory retrieval by different criteria"""
        # Add test memories
        self.memory_system.add_memory(
            content="Important task",
            memory_type=MemoryType.LONG_TERM,
            emotional_tags=[EmotionalTag.IMPORTANT],
            context_tags=["task"]
        )
        
        self.memory_system.add_memory(
            content="Urgent meeting",
            memory_type=MemoryType.LONG_TERM,
            emotional_tags=[EmotionalTag.URGENT],
            context_tags=["meeting"]
        )
        
        # Retrieve by context
        task_memories = self.memory_system.retrieve_memory(
            context_tags=["task"]
        )
        self.assertEqual(len(task_memories), 1)
        self.assertEqual(task_memories[0].content, "Important task")
        
        # Retrieve by emotional tag
        urgent_memories = self.memory_system.retrieve_memory(
            emotional_tags=[EmotionalTag.URGENT]
        )
        self.assertEqual(len(urgent_memories), 1)
        self.assertEqual(urgent_memories[0].content, "Urgent meeting")
        
    def test_memory_consolidation(self):
        """Test memory consolidation process"""
        # Add memories with different importance
        for i in range(15):  # Exceed working memory capacity
            self.memory_system.add_memory(
                content=f"Memory {i}",
                memory_type=MemoryType.WORKING,
                emotional_tags=[EmotionalTag.NEUTRAL],
                context_tags=["test"]
            )
            
        # Wait for consolidation
        import time
        time.sleep(1)
        
        # Verify working memory size
        self.assertLessEqual(
            len(self.memory_system.working_memory.items),
            10  # Working memory capacity
        )
        
    def test_emotional_processing(self):
        """Test emotional processing and importance calculation"""
        # Add memory with multiple emotional tags
        memory_id = self.memory_system.add_memory(
            content="Critical emotional memory",
            memory_type=MemoryType.EMOTIONAL,
            emotional_tags=[EmotionalTag.URGENT, EmotionalTag.IMPORTANT],
            context_tags=["test"]
        )
        
        # Retrieve and check importance
        memory = self.memory_system.long_term_store.memory_index[memory_id]
        self.assertGreater(memory.importance_score, 0.8)  # High importance
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

class TestSymbolicEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SymbolicEngine()
        self.setup_test_rules()
        
    def setup_test_rules(self):
        """Setup test rules for the engine"""
        # Pattern matching rule
        def pattern_action(memory: MemoryItem) -> MemoryItem:
            memory.metadata['pattern_matched'] = True
            return memory
            
        self.engine.add_rule(
            name="test_pattern",
            rule_type=RuleType.PATTERN,
            pattern=r"important|urgent",
            action=pattern_action,
            priority=1
        )
        
        # Transformation rule
        def transform_action(memory: MemoryItem) -> MemoryItem:
            memory.content = memory.content.upper()
            return memory
            
        self.engine.add_rule(
            name="test_transform",
            rule_type=RuleType.TRANSFORMATION,
            pattern=r".*",
            action=transform_action,
            priority=2
        )
        
        # Inference rule
        def inference_condition(memory: MemoryItem) -> bool:
            return EmotionalTag.URGENT in memory.emotional_tags
            
        def inference_action(memory: MemoryItem) -> MemoryItem:
            memory.context_tags.append("high_priority")
            return memory
            
        self.engine.add_rule(
            name="test_inference",
            rule_type=RuleType.INFERENCE,
            pattern=r".*",
            action=inference_action,
            conditions=[inference_condition],
            priority=3
        )
        
    def test_pattern_matching(self):
        """Test pattern matching rules"""
        memory = MemoryItem(
            content="This is an important test",
            created_at=datetime.now(),
            memory_type=MemoryType.WORKING,
            last_accessed=datetime.now(),
            access_count=1,
            emotional_tags=[EmotionalTag.NEUTRAL],
            importance_score=0.5,
            context_tags=[],
            related_memories=[],
            metadata={}
        )
        
        processed = self.engine.process_memory(memory)
        self.assertTrue(processed.metadata.get('pattern_matched'))
        
    def test_transformation(self):
        """Test transformation rules"""
        memory = MemoryItem(
            content="Transform this text",
            created_at=datetime.now(),
            memory_type=MemoryType.WORKING,
            last_accessed=datetime.now(),
            access_count=1,
            emotional_tags=[EmotionalTag.NEUTRAL],
            importance_score=0.5,
            context_tags=[],
            related_memories=[],
            metadata={}
        )
        
        processed = self.engine.process_memory(memory)
        self.assertEqual(processed.content, "TRANSFORM THIS TEXT")
        
    def test_inference(self):
        """Test inference rules"""
        memory = MemoryItem(
            content="Urgent task",
            created_at=datetime.now(),
            memory_type=MemoryType.WORKING,
            last_accessed=datetime.now(),
            access_count=1,
            emotional_tags=[EmotionalTag.URGENT],
            importance_score=0.5,
            context_tags=[],
            related_memories=[],
            metadata={}
        )
        
        processed = self.engine.process_memory(memory)
        self.assertIn("high_priority", processed.context_tags)
        
    def test_rule_builder(self):
        """Test rule builder utility"""
        pattern_rule = RuleBuilder.create_pattern_rule(
            pattern=r"test",
            action=lambda x: x,
            priority=1
        )
        
        self.assertEqual(pattern_rule.rule_type, RuleType.PATTERN)
        self.assertEqual(pattern_rule.priority, 1)
        
        transform_rule = RuleBuilder.create_transformation_rule(
            pattern=r".*",
            transform_func=lambda x: x,
            priority=2
        )
        
        self.assertEqual(transform_rule.rule_type, RuleType.TRANSFORMATION)
        self.assertEqual(transform_rule.priority, 2)

if __name__ == '__main__':
    unittest.main() 