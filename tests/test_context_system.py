import unittest
from datetime import datetime, timedelta
from core.memory.context_system import (
    TriggerType,
    ContextTrigger,
    ContextMatcher,
    ContextualTriggerSystem,
    ContextAnalyzer,
    ContextBuilder
)
from core.memory.memory_system import MemoryItem, EmotionalTag

class TestContextSystem(unittest.TestCase):
    def setUp(self):
        self.trigger_system = ContextualTriggerSystem()
        self.analyzer = ContextAnalyzer(self.trigger_system)
        
    def create_test_memory(self):
        return MemoryItem(
            content="Test memory about AI and machine learning",
            memory_type="working",
            context_tags=["AI", "learning"],
            emotional_tags=[EmotionalTag.NEUTRAL],
            importance_score=0.8,
            last_accessed=datetime.now()
        )
        
    def test_semantic_trigger(self):
        # Test semantic trigger creation and evaluation
        trigger = ContextBuilder.create_semantic_trigger(
            pattern="artificial intelligence and learning",
            threshold=0.7
        )
        self.trigger_system.add_trigger("ai_context", TriggerType.SEMANTIC, trigger.pattern)
        
        memory = self.create_test_memory()
        contexts = self.trigger_system.find_relevant_contexts(
            memory,
            "Discussion about AI and neural networks"
        )
        self.assertIn("ai_context", contexts)
        
    def test_structural_trigger(self):
        # Test structural pattern matching
        trigger = ContextBuilder.create_structural_trigger(
            pattern="AI",
            conditions=["learning", "neural"]
        )
        self.trigger_system.add_trigger("ai_structural", TriggerType.STRUCTURAL, 
                                      trigger.pattern, trigger.conditions)
        
        memory = self.create_test_memory()
        contexts = self.trigger_system.find_relevant_contexts(
            memory,
            "Some context"
        )
        self.assertIn("ai_structural", contexts)
        
    def test_temporal_trigger(self):
        # Test temporal relevance
        trigger = ContextTrigger(
            trigger_type=TriggerType.TEMPORAL,
            pattern="recent",
            conditions=[],
            activation_threshold=0.5,
            metadata={}
        )
        self.trigger_system.add_trigger("temporal", TriggerType.TEMPORAL, trigger.pattern)
        
        # Test with recent memory
        memory = self.create_test_memory()
        contexts = self.trigger_system.find_relevant_contexts(
            memory,
            "current context"
        )
        self.assertIn("temporal", contexts)
        
        # Test with old memory
        memory.last_accessed = datetime.now() - timedelta(days=7)
        contexts = self.trigger_system.find_relevant_contexts(
            memory,
            "current context"
        )
        self.assertNotIn("temporal", contexts)
        
    def test_emotional_trigger(self):
        # Test emotional trigger
        trigger = ContextTrigger(
            trigger_type=TriggerType.EMOTIONAL,
            pattern=EmotionalTag.NEUTRAL.value,
            conditions=[],
            activation_threshold=0.5,
            metadata={}
        )
        self.trigger_system.add_trigger("emotional", TriggerType.EMOTIONAL, trigger.pattern)
        
        memory = self.create_test_memory()
        contexts = self.trigger_system.find_relevant_contexts(
            memory,
            "current context"
        )
        self.assertIn("emotional", contexts)
        
    def test_composite_trigger(self):
        # Test composite trigger
        trigger = ContextBuilder.create_composite_trigger(
            pattern="AI research",
            conditions=["learning", "neural networks"],
            threshold=0.6
        )
        self.trigger_system.add_trigger("composite", TriggerType.COMPOSITE, 
                                      trigger.pattern, trigger.conditions)
        
        memory = self.create_test_memory()
        contexts = self.trigger_system.find_relevant_contexts(
            memory,
            "Research in AI and neural networks"
        )
        self.assertIn("composite", contexts)
        
    def test_context_analyzer(self):
        # Test analyzer functionality
        trigger = ContextBuilder.create_semantic_trigger("AI research")
        self.trigger_system.add_trigger("ai_research", TriggerType.SEMANTIC, trigger.pattern)
        
        memory = self.create_test_memory()
        self.trigger_system.find_relevant_contexts(memory, "AI research context")
        
        # Test pattern analysis
        patterns = self.analyzer.analyze_context_patterns()
        self.assertIn("ai_research", patterns)
        self.assertEqual(patterns["ai_research"]["trigger_type"], "semantic")
        
        # Test statistics
        stats = self.analyzer.get_context_statistics()
        self.assertEqual(stats["total_triggers"], 1)
        self.assertEqual(stats["trigger_types"][TriggerType.SEMANTIC], 1)
        
    def test_context_relationships(self):
        # Test finding related contexts
        self.trigger_system.add_trigger("ai", TriggerType.SEMANTIC, "artificial intelligence")
        self.trigger_system.add_trigger("ml", TriggerType.SEMANTIC, "machine learning")
        
        related = self.analyzer.find_related_contexts("ai")
        self.assertIn("ml", related)

if __name__ == '__main__':
    unittest.main() 