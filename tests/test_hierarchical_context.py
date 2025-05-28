import unittest
from datetime import datetime, timedelta
from core.memory.hierarchical_context import (
    ContextNode,
    HierarchicalContextSystem,
    ContextInheritanceManager
)
from core.memory.context_system import ContextTrigger, TriggerType

class TestHierarchicalContext(unittest.TestCase):
    def setUp(self):
        self.hierarchy = HierarchicalContextSystem()
        self.inheritance_manager = ContextInheritanceManager(self.hierarchy)
        
    def create_test_trigger(self, name: str) -> ContextTrigger:
        return ContextTrigger(
            trigger_type=TriggerType.SEMANTIC,
            pattern=name,
            conditions=[],
            activation_threshold=0.7,
            metadata={"test": True}
        )
        
    def test_context_node_creation(self):
        # Test basic node creation
        trigger = self.create_test_trigger("test")
        node = ContextNode(name="test", trigger=trigger)
        
        self.assertEqual(node.name, "test")
        self.assertEqual(node.trigger, trigger)
        self.assertIsNone(node.parent)
        self.assertEqual(len(node.children), 0)
        
    def test_hierarchy_creation(self):
        # Test hierarchy initialization
        self.assertIn("root", self.hierarchy.nodes)
        root = self.hierarchy.nodes["root"]
        self.assertEqual(root.name, "root")
        self.assertEqual(len(root.children), 0)
        
    def test_add_context(self):
        # Test adding contexts
        trigger = self.create_test_trigger("parent")
        parent = self.hierarchy.add_context("parent", trigger)
        
        self.assertIn("parent", self.hierarchy.nodes)
        self.assertEqual(parent.parent, self.hierarchy.root)
        
        # Add child context
        child_trigger = self.create_test_trigger("child")
        child = self.hierarchy.add_context("child", child_trigger, "parent")
        
        self.assertIn("child", self.hierarchy.nodes)
        self.assertEqual(child.parent, parent)
        self.assertIn(child, parent.children)
        
    def test_remove_context(self):
        # Test removing contexts
        trigger = self.create_test_trigger("parent")
        self.hierarchy.add_context("parent", trigger)
        child_trigger = self.create_test_trigger("child")
        self.hierarchy.add_context("child", child_trigger, "parent")
        
        # Remove parent (should also remove child)
        self.hierarchy.remove_context("parent")
        
        self.assertNotIn("parent", self.hierarchy.nodes)
        self.assertNotIn("child", self.hierarchy.nodes)
        
    def test_context_path(self):
        # Test getting context paths
        trigger = self.create_test_trigger("parent")
        self.hierarchy.add_context("parent", trigger)
        child_trigger = self.create_test_trigger("child")
        self.hierarchy.add_context("child", child_trigger, "parent")
        
        path = self.hierarchy.get_context_path("child")
        self.assertEqual(path, ["root", "parent", "child"])
        
    def test_common_ancestor(self):
        # Test finding common ancestors
        trigger1 = self.create_test_trigger("context1")
        trigger2 = self.create_test_trigger("context2")
        parent = self.hierarchy.add_context("parent", trigger1)
        self.hierarchy.add_context("child1", trigger2, "parent")
        self.hierarchy.add_context("child2", trigger2, "parent")
        
        ancestor = self.hierarchy.find_common_ancestor("child1", "child2")
        self.assertEqual(ancestor, parent)
        
    def test_context_subtree(self):
        # Test getting context subtrees
        trigger = self.create_test_trigger("parent")
        self.hierarchy.add_context("parent", trigger)
        child_trigger = self.create_test_trigger("child")
        self.hierarchy.add_context("child1", child_trigger, "parent")
        self.hierarchy.add_context("child2", child_trigger, "parent")
        
        subtree = self.hierarchy.get_context_subtree("parent")
        self.assertEqual(subtree, {"parent", "child1", "child2"})
        
    def test_context_activation(self):
        # Test context activation
        trigger = self.create_test_trigger("test")
        self.hierarchy.add_context("test", trigger)
        
        # Test initial state
        self.assertEqual(len(self.hierarchy.get_active_contexts()), 0)
        
        # Activate context
        self.hierarchy.update_context_activation("test")
        active = self.hierarchy.get_active_contexts()
        self.assertIn("test", active)
        
        # Test activation decay
        node = self.hierarchy.nodes["test"]
        node.last_activated = datetime.now() - timedelta(hours=1)
        active = self.hierarchy.get_active_contexts(threshold=0.5)  # 30 minutes
        self.assertNotIn("test", active)
        
    def test_inheritance(self):
        # Test trigger inheritance
        parent_trigger = self.create_test_trigger("parent")
        child_trigger = self.create_test_trigger("child")
        
        self.hierarchy.add_context("parent", parent_trigger)
        self.hierarchy.add_context("child", child_trigger, "parent")
        
        # Test inherited triggers
        triggers = self.inheritance_manager.get_inherited_triggers("child")
        self.assertEqual(len(triggers), 3)  # child, parent, root
        self.assertEqual(triggers[0], child_trigger)
        
        # Test inherited metadata
        parent_metadata = {"key1": "parent_value", "shared": "parent"}
        child_metadata = {"key2": "child_value", "shared": "child"}
        
        parent = self.hierarchy.nodes["parent"]
        child = self.hierarchy.nodes["child"]
        parent.metadata = parent_metadata
        child.metadata = child_metadata
        
        inherited = self.inheritance_manager.get_inherited_metadata("child")
        self.assertEqual(inherited["key1"], "parent_value")
        self.assertEqual(inherited["key2"], "child_value")
        self.assertEqual(inherited["shared"], "child")  # Child value takes precedence
        
    def test_activation_propagation(self):
        # Test activation propagation
        trigger = self.create_test_trigger("parent")
        self.hierarchy.add_context("parent", trigger)
        child_trigger = self.create_test_trigger("child")
        self.hierarchy.add_context("child", child_trigger, "parent")
        
        # Activate child and test propagation
        self.inheritance_manager.propagate_activation("child")
        
        # Both child and parent should be active
        active = self.hierarchy.get_active_contexts()
        self.assertIn("child", active)
        self.assertIn("parent", active)

if __name__ == '__main__':
    unittest.main() 