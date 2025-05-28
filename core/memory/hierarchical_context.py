from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from .context_system import ContextualTrigger

class ContextNode:
    """Represents a node in the context hierarchy"""
    def __init__(self,
                 name: str,
                 trigger: ContextualTrigger,
                 parent: Optional['ContextNode'] = None):
        self.name = name
        self.trigger = trigger
        self.parent = parent
        self.children: List[ContextNode] = []
        self.activation_count = 0
        self.last_activated = None
        self.metadata: Dict[str, Any] = {}
        
    def add_child(self, child: 'ContextNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        
    def remove_child(self, child: 'ContextNode'):
        """Remove a child node"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            
    def get_ancestors(self) -> List['ContextNode']:
        """Get all ancestor nodes"""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors
        
    def get_descendants(self) -> List['ContextNode']:
        """Get all descendant nodes"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants
        
    def get_siblings(self) -> List['ContextNode']:
        """Get all sibling nodes"""
        if not self.parent:
            return []
        return [child for child in self.parent.children if child != self]
        
    def activate(self):
        """Activate this context node"""
        self.activation_count += 1
        self.last_activated = datetime.now()

class HierarchicalContextSystem:
    """Manages hierarchical context relationships"""
    def __init__(self, trigger_system=None):
        self.nodes: Dict[str, ContextNode] = {}
        self.root: Optional[ContextNode] = None
        self.active_contexts: Set[str] = set()
        self.trigger_system = trigger_system
        if not self.trigger_system:
            from .context_system import ContextualTriggerSystem
            self.trigger_system = ContextualTriggerSystem()
            
    def add_context(self,
                   name: str,
                   trigger: ContextualTrigger,
                   parent_name: Optional[str] = None) -> ContextNode:
        """Add a new context to the hierarchy"""
        if name in self.nodes:
            raise ValueError(f"Context '{name}' already exists")
            
        # Create node and add to hierarchy
        node = ContextNode(name, trigger)
        self.nodes[name] = node
        
        if parent_name:
            parent = self.nodes.get(parent_name)
            if not parent:
                raise ValueError(f"Parent context '{parent_name}' not found")
            parent.add_child(node)
        elif not self.root:
            self.root = node
            
        # Add trigger to trigger system
        if self.trigger_system:
            self.trigger_system.add_trigger(
                name,
                trigger.trigger_type,
                trigger.condition,
                trigger.threshold
            )
            
            # Add hierarchical relationship triggers
            if parent_name:
                parent_trigger = self.nodes[parent_name].trigger
                self.trigger_system.add_trigger(
                    f"{parent_name}_to_{name}",
                    parent_trigger.trigger_type,
                    parent_trigger.condition,
                    threshold=0.6  # Lower threshold for hierarchical relationships
                )
                
        return node
        
    def remove_context(self, name: str):
        """Remove a context from the hierarchy"""
        if name not in self.nodes:
            return
            
        node = self.nodes[name]
        if node.parent:
            node.parent.remove_child(node)
            
        # Reassign children to parent
        if node.parent and node.children:
            for child in node.children:
                node.parent.add_child(child)
                
        del self.nodes[name]
        if self.root and self.root.name == name:
            self.root = None
            
    def activate_context(self, name: str):
        """Activate a context and its ancestors"""
        if name not in self.nodes:
            return
            
        node = self.nodes[name]
        node.activate()
        self.active_contexts.add(name)
        
        # Activate ancestors with decreasing threshold
        threshold = 0.9
        current = node.parent
        while current and threshold >= 0.5:
            current.activate()
            self.active_contexts.add(current.name)
            # Evaluate trigger with current threshold
            if self.trigger_system:
                relevance = self.trigger_system.evaluate_trigger(current.name, name)
                if relevance < threshold:
                    break
            threshold *= 0.8  # Decrease threshold for each level
            current = current.parent
            
    def deactivate_context(self, name: str):
        """Deactivate a context and its descendants"""
        if name not in self.nodes:
            return
            
        node = self.nodes[name]
        self.active_contexts.discard(name)
        
        # Deactivate descendants
        for descendant in node.get_descendants():
            self.active_contexts.discard(descendant.name)
            
    def get_active_contexts(self) -> Set[str]:
        """Get all currently active contexts"""
        return self.active_contexts.copy()
        
    def get_context_path(self, name: str) -> List[str]:
        """Get path from root to context"""
        if name not in self.nodes:
            return []
            
        path = []
        current = self.nodes[name]
        while current:
            path.append(current.name)
            current = current.parent
        return list(reversed(path))
        
    def find_common_ancestor(self, name1: str, name2: str) -> Optional[ContextNode]:
        """Find the lowest common ancestor of two contexts"""
        node1 = self.nodes.get(name1)
        node2 = self.nodes.get(name2)
        if not (node1 and node2):
            return None
            
        ancestors1 = set([node1] + node1.get_ancestors())
        current = node2
        while current:
            if current in ancestors1:
                return current
            current = current.parent
        return None
        
    def get_context_subtree(self, name: str) -> Set[str]:
        """Get all contexts in the subtree rooted at the specified context"""
        node = self.nodes.get(name)
        if not node:
            raise ValueError(f"Context '{name}' not found")
            
        subtree = {node.name}
        for descendant in node.get_descendants():
            subtree.add(descendant.name)
        return subtree
        
    def update_context_activation(self, name: str):
        """Update activation statistics for a context and its ancestors"""
        node = self.nodes.get(name)
        if not node:
            raise ValueError(f"Context '{name}' not found")
            
        current = node
        while current:
            current.last_activated = datetime.now()
            current.activation_count += 1
            current = current.parent
            
class ContextInheritanceManager:
    """Manages inheritance of properties and triggers in the context hierarchy"""
    def __init__(self, hierarchy: HierarchicalContextSystem):
        self.hierarchy = hierarchy
        
    def get_inherited_triggers(self, context_name: str) -> List[ContextualTrigger]:
        """Get all triggers that apply to a context (including inherited)"""
        node = self.hierarchy.nodes.get(context_name)
        if not node:
            raise ValueError(f"Context '{context_name}' not found")
            
        triggers = [node.trigger]
        current = node.parent
        while current:
            triggers.append(current.trigger)
            current = current.parent
        return triggers
        
    def get_inherited_metadata(self, context_name: str) -> Dict[str, Any]:
        """Get combined metadata from context and its ancestors"""
        node = self.hierarchy.nodes.get(context_name)
        if not node:
            raise ValueError(f"Context '{context_name}' not found")
            
        metadata = {}
        current = node
        while current:
            # Update metadata with current node's metadata (earlier values take precedence)
            metadata.update(current.metadata)
            current = current.parent
        return metadata
        
    def propagate_activation(self, context_name: str, activation_threshold: float = 0.5):
        """Propagate context activation to related contexts"""
        node = self.hierarchy.nodes.get(context_name)
        if not node:
            raise ValueError(f"Context '{context_name}' not found")
            
        # Activate ancestors with decreasing strength
        strength = 1.0
        current = node.parent
        while current and strength >= activation_threshold:
            self.hierarchy.update_context_activation(current.name)
            strength *= 0.7  # Decay factor
            current = current.parent
            
        # Activate children with decreasing strength
        for child in node.children:
            if child.trigger.activation_threshold <= activation_threshold:
                self.hierarchy.update_context_activation(child.name) 