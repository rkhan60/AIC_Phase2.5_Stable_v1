import time
import unittest
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict
import logging
from core.memory.memory_system import MemoryItem, MemoryType, EmotionalTag
from core.memory.context_system import ContextualTriggerSystem, TriggerType
from core.memory.hierarchical_context import HierarchicalContextSystem
from core.memory.context_aware_consolidation import ContextAwareConsolidation
from core.memory.context_optimizer import ContextOptimizer
from core.visualization.context_visualizer import ContextVisualizer
from core.memory.context_system import ContextualTrigger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemDiagnostics:
    """Comprehensive diagnostics for the memory and context system"""
    
    def __init__(self):
        self.context_system = ContextualTriggerSystem()
        self.hierarchy = HierarchicalContextSystem(trigger_system=self.context_system)
        self.consolidation = ContextAwareConsolidation(
            self.context_system,
            self.hierarchy
        )
        self.optimizer = ContextOptimizer(
            self.hierarchy,
            self.consolidation
        )
        self.visualizer = ContextVisualizer(self.hierarchy)
        self.test_memories: List[MemoryItem] = []
        
    def setup_test_data(self):
        """Initialize test data"""
        logger.info("Setting up test data...")
        
        # Create test contexts
        contexts = {
            "work": ["meetings", "projects", "tasks"],
            "personal": ["hobbies", "health", "finance"],
            "learning": ["courses", "research", "practice"]
        }
        
        for domain, subcontexts in contexts.items():
            self.hierarchy.add_context(domain, self._create_trigger(domain))
            for sub in subcontexts:
                self.hierarchy.add_context(sub, self._create_trigger(sub), domain)
                
        # Create test memories
        self.test_memories = [
            self._create_memory(
                "Important project meeting tomorrow",
                ["work", "meetings"],
                EmotionalTag.NEUTRAL
            ),
            self._create_memory(
                "New research findings on memory systems",
                ["learning", "research"],
                EmotionalTag.POSITIVE
            ),
            self._create_memory(
                "Weekly fitness goals achieved",
                ["personal", "health"],
                EmotionalTag.POSITIVE
            )
        ]
        
        logger.info("Test data setup completed")
        
    def _create_trigger(self, pattern: str):
        """Helper to create test triggers"""
        return ContextualTrigger(
            pattern=pattern,
            trigger_type=TriggerType.SEMANTIC,
            condition=pattern,
            threshold=0.7
        )
        
    def _create_memory(self,
                      content: str,
                      context_tags: List[str],
                      emotional_tag: EmotionalTag) -> MemoryItem:
        """Helper to create test memories"""
        return MemoryItem(
            content=content,
            memory_type=MemoryType.WORKING,
            context_tags=context_tags,
            emotional_tags=[emotional_tag],
            importance_score=0.7,
            last_accessed=datetime.now()
        )
        
    def run_performance_diagnostics(self):
        """Test system performance"""
        logger.info("Running performance diagnostics...")
        
        metrics = {
            "retrieval_times": [],
            "consolidation_times": [],
            "context_switch_times": []
        }
        
        # Test retrieval performance
        for _ in range(10):
            start = time.time()
            self.optimizer.optimize_retrieval(
                self.test_memories,
                "work"
            )
            metrics["retrieval_times"].append(time.time() - start)
            
        # Test consolidation performance
        for memory in self.test_memories:
            start = time.time()
            self.consolidation.consolidate_memory(
                memory,
                "work"
            )
            metrics["consolidation_times"].append(time.time() - start)
            
        # Test context switching
        contexts = ["work", "personal", "learning"]
        for i in range(len(contexts) - 1):
            start = time.time()
            self.optimizer.optimize_retrieval(
                self.test_memories,
                contexts[i+1]
            )
            metrics["context_switch_times"].append(time.time() - start)
            
        self._report_performance_metrics(metrics)
        
    def _report_performance_metrics(self, metrics: Dict[str, List[float]]):
        """Report performance metrics"""
        logger.info("\nPerformance Metrics:")
        for operation, times in metrics.items():
            logger.info(f"{operation}:")
            logger.info(f"  Average: {np.mean(times):.4f}s")
            logger.info(f"  Min: {np.min(times):.4f}s")
            logger.info(f"  Max: {np.max(times):.4f}s")
            
    def run_reliability_diagnostics(self):
        """Test system reliability"""
        logger.info("\nRunning reliability diagnostics...")
        
        # Test context hierarchy integrity
        self._test_hierarchy_integrity()
        
        # Test memory consolidation reliability
        self._test_consolidation_reliability()
        
        # Test context switching reliability
        self._test_context_switching_reliability()
        
        # Test optimization stability
        self._test_optimization_stability()
        
    def _test_hierarchy_integrity(self):
        """Test context hierarchy integrity"""
        logger.info("Testing hierarchy integrity...")
        
        # Verify all nodes are properly connected
        for name, node in self.hierarchy.nodes.items():
            if node.parent:
                assert name in [child.name for child in node.parent.children]
                
        # Verify no circular references
        for name, node in self.hierarchy.nodes.items():
            ancestors = node.get_ancestors()
            assert node not in ancestors
            
        logger.info("Hierarchy integrity verified")
        
    def _test_consolidation_reliability(self):
        """Test memory consolidation reliability"""
        logger.info("Testing consolidation reliability...")
        
        for memory in self.test_memories:
            # Test multiple consolidations
            for _ in range(3):
                consolidated = self.consolidation.consolidate_memory(
                    memory,
                    "work"
                )
                assert consolidated is not None
                assert consolidated.importance_score >= memory.importance_score
                
        logger.info("Consolidation reliability verified")
        
    def _test_context_switching_reliability(self):
        """Test context switching reliability"""
        logger.info("Testing context switching reliability...")
        
        contexts = ["work", "personal", "learning"]
        last_results = None
        
        # Test consistent results across switches
        for context in contexts * 2:
            results = self.optimizer.optimize_retrieval(
                self.test_memories,
                context
            )
            if last_results and context == contexts[0]:
                assert len(results) == len(last_results)
            last_results = results
            
        logger.info("Context switching reliability verified")
        
    def _test_optimization_stability(self):
        """Test optimization stability"""
        logger.info("Testing optimization stability...")
        
        # Test cache stability
        initial_size = len(self.optimizer.context_cache)
        for _ in range(10):
            self.optimizer.optimize_retrieval(
                self.test_memories,
                "work"
            )
        assert len(self.optimizer.context_cache) >= initial_size
        
        # Test recommendations stability
        recommendations = self.optimizer.optimize_context_structure()
        assert isinstance(recommendations, list)
        
        logger.info("Optimization stability verified")
        
    def run_integration_diagnostics(self):
        """Test system integration"""
        logger.info("\nRunning integration diagnostics...")
        
        # Test visualization integration
        self._test_visualization_integration()
        
        # Test optimization integration
        self._test_optimization_integration()
        
        # Test end-to-end workflow
        self._test_end_to_end_workflow()
        
    def _test_visualization_integration(self):
        """Test visualization system integration"""
        logger.info("Testing visualization integration...")
        
        # Test graph building
        self.visualizer.build_graph()
        assert len(self.visualizer.graph.nodes) == len(self.hierarchy.nodes)
        
        # Test report generation
        report = self.visualizer.generate_context_report()
        assert isinstance(report, str)
        assert len(report) > 0
        
        logger.info("Visualization integration verified")
        
    def _test_optimization_integration(self):
        """Test optimization system integration"""
        logger.info("Testing optimization integration...")
        
        # Test metrics integration
        self.optimizer.optimize_retrieval(
            self.test_memories,
            "work"
        )
        report = self.optimizer.get_optimization_report()
        assert isinstance(report, dict)
        assert "most_accessed_contexts" in report
        
        logger.info("Optimization integration verified")
        
    def _test_end_to_end_workflow(self):
        """Test complete system workflow"""
        logger.info("Testing end-to-end workflow...")
        
        # Create new memory
        memory = self._create_memory(
            "Testing system integration",
            ["work", "tasks"],
            EmotionalTag.NEUTRAL
        )
        
        # Process through system
        consolidated = self.consolidation.consolidate_memory(
            memory,
            "work"
        )
        assert consolidated is not None
        
        optimized = self.optimizer.optimize_retrieval(
            [consolidated],
            "work"
        )
        assert len(optimized) > 0
        
        logger.info("End-to-end workflow verified")
        
    def run_all_diagnostics(self):
        """Run all diagnostic tests"""
        logger.info("Starting comprehensive system diagnostics...")
        
        try:
            self.setup_test_data()
            self.run_performance_diagnostics()
            self.run_reliability_diagnostics()
            self.run_integration_diagnostics()
            logger.info("All diagnostics completed successfully")
            
        except Exception as e:
            logger.error(f"Diagnostics failed: {str(e)}")
            raise

if __name__ == "__main__":
    diagnostics = SystemDiagnostics()
    diagnostics.run_all_diagnostics() 