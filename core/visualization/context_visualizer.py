import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from ..memory.hierarchical_context import HierarchicalContextSystem, ContextNode
import json

class ContextVisualizer:
    """Visualizes context hierarchies and relationships"""
    def __init__(self, hierarchy: HierarchicalContextSystem):
        self.hierarchy = hierarchy
        self.graph = nx.DiGraph()
        
    def build_graph(self):
        """Build networkx graph from context hierarchy"""
        self.graph.clear()
        
        # Add nodes
        for name, node in self.hierarchy.nodes.items():
            self.graph.add_node(
                name,
                activation_count=node.activation_count,
                last_activated=node.last_activated,
                metadata=node.metadata
            )
            
        # Add edges
        for name, node in self.hierarchy.nodes.items():
            if node.parent:
                self.graph.add_edge(node.parent.name, name)
                
    def plot_hierarchy(self, 
                      figsize: tuple = (12, 8),
                      node_size: int = 1000,
                      with_labels: bool = True,
                      font_size: int = 10):
        """Plot the context hierarchy"""
        self.build_graph()
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        active_contexts = self.hierarchy.get_active_contexts()
        node_colors = ['red' if node in active_contexts else 'lightblue' 
                      for node in self.graph.nodes()]
        
        nx.draw_networkx_nodes(self.graph, pos,
                             node_color=node_colors,
                             node_size=node_size)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, 
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20)
        
        if with_labels:
            nx.draw_networkx_labels(self.graph, pos, font_size=font_size)
            
        plt.title("Context Hierarchy Visualization")
        plt.axis('off')
        
    def plot_activation_heatmap(self, figsize: tuple = (10, 6)):
        """Plot context activation heatmap"""
        activations = {name: node.activation_count 
                      for name, node in self.hierarchy.nodes.items()}
        
        plt.figure(figsize=figsize)
        plt.bar(activations.keys(), activations.values())
        plt.xticks(rotation=45)
        plt.title("Context Activation Heatmap")
        plt.xlabel("Context")
        plt.ylabel("Activation Count")
        
    def export_graph_json(self) -> Dict[str, Any]:
        """Export graph structure as JSON"""
        self.build_graph()
        
        data = {
            "nodes": [],
            "edges": []
        }
        
        # Export nodes
        for node in self.graph.nodes(data=True):
            node_data = {
                "id": node[0],
                "metadata": node[1]
            }
            data["nodes"].append(node_data)
            
        # Export edges
        for edge in self.graph.edges():
            edge_data = {
                "source": edge[0],
                "target": edge[1]
            }
            data["edges"].append(edge_data)
            
        return data
        
    def generate_context_report(self) -> str:
        """Generate a detailed report of the context hierarchy"""
        report = []
        report.append("Context Hierarchy Analysis Report")
        report.append("=" * 30)
        
        # Basic statistics
        report.append("\nBasic Statistics:")
        report.append(f"Total Contexts: {len(self.hierarchy.nodes)}")
        active = len(self.hierarchy.get_active_contexts())
        report.append(f"Active Contexts: {active}")
        
        # Context tree
        report.append("\nContext Tree:")
        self._append_tree_representation(self.hierarchy.root, report)
        
        # Activation statistics
        report.append("\nActivation Statistics:")
        for name, node in self.hierarchy.nodes.items():
            if node.activation_count > 0:
                report.append(f"{name}: {node.activation_count} activations")
                
        # Metadata analysis
        report.append("\nMetadata Analysis:")
        for name, node in self.hierarchy.nodes.items():
            if node.metadata:
                report.append(f"{name}: {json.dumps(node.metadata, indent=2)}")
                
        return "\n".join(report)
        
    def _append_tree_representation(self, 
                                  node: ContextNode,
                                  report: list,
                                  level: int = 0):
        """Helper method to build tree representation"""
        indent = "  " * level
        report.append(f"{indent}├── {node.name}")
        
        for child in node.children:
            self._append_tree_representation(child, report, level + 1)
            
    def plot_context_relationships(self, context_name: str, figsize: tuple = (10, 6)):
        """Plot relationships for a specific context"""
        node = self.hierarchy.nodes.get(context_name)
        if not node:
            raise ValueError(f"Context '{context_name}' not found")
            
        plt.figure(figsize=figsize)
        
        # Create a subgraph with related contexts
        related = nx.DiGraph()
        
        # Add ancestors
        ancestors = node.get_ancestors()
        for ancestor in ancestors:
            related.add_node(ancestor.name, type="ancestor")
            if ancestor.parent:
                related.add_edge(ancestor.parent.name, ancestor.name)
                
        # Add descendants
        descendants = node.get_descendants()
        for descendant in descendants:
            related.add_node(descendant.name, type="descendant")
            if descendant.parent:
                related.add_edge(descendant.parent.name, descendant.name)
                
        # Add siblings
        siblings = node.get_siblings()
        for sibling in siblings:
            related.add_node(sibling.name, type="sibling")
            if sibling.parent:
                related.add_edge(sibling.parent.name, sibling.name)
                
        # Add the node itself
        related.add_node(node.name, type="focus")
        if node.parent:
            related.add_edge(node.parent.name, node.name)
            
        # Draw the graph
        pos = nx.spring_layout(related)
        
        # Draw nodes with different colors based on relationship
        colors = {
            "ancestor": "lightblue",
            "descendant": "lightgreen",
            "sibling": "lightgray",
            "focus": "red"
        }
        
        for node_type in colors:
            nodes = [n for n, attr in related.nodes(data=True)
                    if attr.get("type") == node_type]
            if nodes:
                nx.draw_networkx_nodes(related, pos,
                                     nodelist=nodes,
                                     node_color=colors[node_type],
                                     node_size=1000)
                
        nx.draw_networkx_edges(related, pos,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20)
        
        nx.draw_networkx_labels(related, pos)
        
        plt.title(f"Context Relationships: {context_name}")
        plt.axis('off') 