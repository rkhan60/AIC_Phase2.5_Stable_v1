import unittest
from datetime import datetime, timedelta
from core.memory.context_system import (
    ContextualTriggerSystem,
    ContextTrigger,
    TriggerType
)
from core.memory.hierarchical_context import HierarchicalContextSystem
from core.memory.context_aware_consolidation import ContextAwareConsolidation
from core.memory.memory_system import MemoryItem, MemoryType, EmotionalTag

class TestContextAwareConsolidation(unittest.TestCase):
    def setUp(self):
        self.context_system = ContextualTriggerSystem()
        self.hierarchy = HierarchicalContextSystem()
        self.consolidation = ContextAwareConsolidation(
            self.context_system,
            self.hierarchy
        )
        
        # Set up test contexts
        self.setup_test_contexts()
        
    def setup_test_contexts(self):
        # Create a test context hierarchy
        # root
        # ├── work
        # │   ├── meetings
        # │   └── projects
        # └── personal
        #     ├── hobbies
        #     └── tasks
        
        contexts = {
            "work": TriggerType.SEMANTIC,
            "meetings": TriggerType.SEMANTIC,
            "projects": TriggerType.SEMANTIC,
            "personal": TriggerType.SEMANTIC,
            "hobbies": TriggerType.SEMANTIC,
            "tasks": TriggerType.SEMANTIC
        }
        
        # Add contexts with their triggers
        for name, trigger_type in contexts.items():
            trigger = ContextTrigger(
                trigger_type=trigger_type,
                pattern=name,
                conditions=[],
                activation_threshold=0.7,
                metadata={"domain": name}
            )
            
            if name in ["meetings", "projects"]:
                self.hierarchy.add_context(name, trigger, "work")
            elif name in ["hobbies", "tasks"]:
                self.hierarchy.add_context(name, trigger, "personal")
            else:
                self.hierarchy.add_context(name, trigger)
                
    def create_test_memory(self, content: str, memory_type: str = "working") -> MemoryItem:
        return MemoryItem(
            content=content,
            memory_type=memory_type,
            context_tags=[],
            emotional_tags=[EmotionalTag.NEUTRAL],
            importance_score=0.5,
            last_accessed=datetime.now()
        )
        
    def test_consolidation_priority(self):
        # Test priority calculation
        memory = self.create_test_memory("Important work meeting tomorrow")
        
        # Activate relevant contexts
        self.hierarchy.update_context_activation("work")
        self.hierarchy.update_context_activation("meetings")
        
        priority = self.consolidation.evaluate_consolidation_priority(
            memory, "Discussion about work meetings")
        
        self.assertGreater(priority, 0.5)
        
    def test_should_consolidate(self):
        # Test consolidation decision
        memory = self.create_test_memory("Personal hobby project")
        memory.memory_type = MemoryType.WORKING
        
        # Activate relevant contexts
        self.hierarchy.update_context_activation("personal")
        self.hierarchy.update_context_activation("hobbies")
        
        should_consolidate = self.consolidation.should_consolidate(
            memory, "Thinking about hobbies")
        
        self.assertTrue(should_consolidate)
        
    def test_consolidation_contexts(self):
        # Test context identification for consolidation
        memory = self.create_test_memory("Project meeting for work")
        
        # Activate some contexts
        self.hierarchy.update_context_activation("work")
        self.hierarchy.update_context_activation("meetings")
        
        contexts = self.consolidation.get_consolidation_contexts(
            memory, "Work project discussion")
        
        self.assertIn("work", contexts)
        self.assertIn("meetings", contexts)
        self.assertIn("projects", contexts)
        
    def test_context_relevance(self):
        # Test context relevance evaluation
        memory = self.create_test_memory("Personal task list")
        memory.context_tags = ["personal", "tasks"]
        
        relevance = self.consolidation.evaluate_context_relevance(
            memory, "personal")
        
        self.assertGreater(relevance, 0.5)
        
    def test_memory_context_update(self):
        # Test memory context updating
        memory = self.create_test_memory("Work project status update")
        
        self.consolidation.update_memory_contexts(
            memory, "Project status meeting")
        
        self.assertIn("work", memory.context_tags)
        self.assertIn("projects", memory.context_tags)
        
    def test_memory_consolidation(self):
        # Test full consolidation process
        memory = self.create_test_memory("Important work meeting notes")
        memory.importance_score = 0.6
        
        # Activate relevant contexts
        self.hierarchy.update_context_activation("work")
        self.hierarchy.update_context_activation("meetings")
        
        consolidated = self.consolidation.consolidate_memory(
            memory, "Work meeting context")
        
        self.assertIsNotNone(consolidated)
        self.assertGreater(consolidated.importance_score, 0.6)
        self.assertIn("work", consolidated.context_tags)
        self.assertIn("meetings", consolidated.context_tags)
        
    def test_related_memories(self):
        # Test finding related memories
        memories = [
            self.create_test_memory("Work project A update"),
            self.create_test_memory("Personal hobby notes"),
            self.create_test_memory("Work project B meeting"),
        ]
        
        # Add context tags
        memories[0].context_tags = ["work", "projects"]
        memories[1].context_tags = ["personal", "hobbies"]
        memories[2].context_tags = ["work", "projects", "meetings"]
        
        related = self.consolidation.get_related_memories("work", memories)
        
        self.assertEqual(len(related), 2)
        self.assertEqual(related[0].content, "Work project B meeting")
        self.assertEqual(related[1].content, "Work project A update")
        
    def test_temporal_relevance(self):
        # Test temporal aspects of consolidation
        memory = self.create_test_memory("Recent work task")
        old_memory = self.create_test_memory("Old work task")
        old_memory.last_accessed = datetime.now() - timedelta(days=2)
        
        # Activate work context
        self.hierarchy.update_context_activation("work")
        
        recent_priority = self.consolidation.evaluate_consolidation_priority(
            memory, "Work context")
        old_priority = self.consolidation.evaluate_consolidation_priority(
            old_memory, "Work context")
        
        self.assertGreater(recent_priority, old_priority)
        
    def test_emotional_relevance(self):
        # Test emotional aspects of consolidation
        memory = self.create_test_memory("Exciting project milestone")
        memory.emotional_tags = [EmotionalTag.POSITIVE]
        
        neutral_memory = self.create_test_memory("Regular project update")
        
        # Activate project context
        self.hierarchy.update_context_activation("projects")
        
        emotional_priority = self.consolidation.evaluate_consolidation_priority(
            memory, "Project context")
        neutral_priority = self.consolidation.evaluate_consolidation_priority(
            neutral_memory, "Project context")
        
        self.assertGreater(emotional_priority, neutral_priority)

if __name__ == '__main__':
    unittest.main() 