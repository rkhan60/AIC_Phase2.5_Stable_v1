import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import json
import tempfile

from core.memory.memory_analytics import MemoryAnalytics, MemoryMetrics, MemoryUsageStats
from core.memory.memory_visualization import MemoryVisualizer
from core.memory.trend_analyzer import AdvancedTrendAnalyzer, TrendInsight

class TestMemoryAnalytics(unittest.TestCase):
    """Integration tests for memory analytics system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.analytics = MemoryAnalytics(Path(self.temp_dir))
        self.visualizer = MemoryVisualizer(self.analytics)
        self.trend_analyzer = AdvancedTrendAnalyzer()
        
        # Generate test data
        self.test_metrics = self._generate_test_metrics()
        self.test_usage_stats = self._generate_test_usage_stats()
        
    def _generate_test_metrics(self) -> List[Dict[str, Any]]:
        """Generate test metrics data"""
        base_time = datetime.now()
        metrics = []
        
        for i in range(100):
            # Generate synthetic metrics with trends and patterns
            timestamp = base_time + timedelta(hours=i)
            
            # Add cyclic pattern
            cycle = np.sin(i * 0.1) * 10
            
            # Add trend
            trend = i * 0.5
            
            # Add noise
            noise = np.random.normal(0, 1)
            
            metrics.append({
                'timestamp': timestamp.isoformat(),
                'total_memories': int(100 + trend + cycle + noise),
                'active_memories': int(50 + trend * 0.5 + cycle + noise),
                'cache_hit_rate': max(0, min(1, 0.7 + np.sin(i * 0.05) * 0.2)),
                'query_latency_ms': 50 + cycle + noise
            })
            
        return metrics
        
    def _generate_test_usage_stats(self) -> List[Dict[str, Any]]:
        """Generate test usage statistics"""
        return [{
            'access_frequency': {f'memory_{i}': i * 2 for i in range(10)},
            'context_distribution': {
                'customer_service': 30,
                'technical_support': 20,
                'sales': 15
            },
            'confidence_distribution': {
                'low': 10,
                'medium': 30,
                'high': 60
            },
            'memory_age_distribution': {
                '1_week': 20,
                '1_month': 40,
                'older': 40
            }
        }]
        
    def test_metrics_collection(self):
        """Test metrics collection and storage"""
        # Add test metrics
        for metric in self.test_metrics:
            self.analytics.metrics_history.append(metric)
            
        # Collect current metrics
        metrics = self.analytics.collect_metrics(None, None)
        
        self.assertIsInstance(metrics, MemoryMetrics)
        self.assertTrue(hasattr(metrics, 'total_memories'))
        self.assertTrue(hasattr(metrics, 'cache_hit_rate'))
        
    def test_usage_pattern_analysis(self):
        """Test usage pattern analysis"""
        # Add test usage stats
        self.analytics.usage_patterns['test'] = self.test_usage_stats
        
        # Analyze patterns
        stats = self.analytics.analyze_usage_patterns(None, None)
        
        self.assertIsInstance(stats, MemoryUsageStats)
        self.assertIn('customer_service', stats.context_distribution)
        self.assertGreater(len(stats.top_accessed_patterns), 0)
        
    def test_trend_analysis(self):
        """Test trend analysis capabilities"""
        # Analyze trends
        insights = self.trend_analyzer.analyze_trends(self.test_metrics)
        
        self.assertIsInstance(insights, dict)
        self.assertGreater(len(insights), 0)
        
        # Check trend detection
        for metric, insight in insights.items():
            self.assertIsInstance(insight, TrendInsight)
            self.assertIn(insight.trend_type, ['increasing', 'decreasing', 'cyclic', 'stable', 'anomaly'])
            self.assertGreaterEqual(insight.confidence, 0)
            self.assertLessEqual(insight.confidence, 1)
            
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        # Add anomaly to test data
        anomaly_metrics = self.test_metrics.copy()
        anomaly_metrics[50]['query_latency_ms'] = 1000  # Obvious anomaly
        
        # Detect anomalies
        anomalies = self.trend_analyzer.detect_anomalies(anomaly_metrics)
        
        self.assertIsInstance(anomalies, dict)
        self.assertIn('query_latency_ms', anomalies)
        self.assertGreater(len(anomalies['query_latency_ms']), 0)
        
    def test_seasonal_pattern_detection(self):
        """Test seasonal pattern detection"""
        patterns = self.trend_analyzer.analyze_seasonal_patterns(
            self.test_metrics,
            'cache_hit_rate'
        )
        
        self.assertIsInstance(patterns, dict)
        self.assertGreater(len(patterns), 0)
        self.assertIn('hourly_pattern', patterns)
        
    def test_visualization_generation(self):
        """Test visualization generation"""
        # Add test data
        for metric in self.test_metrics:
            self.analytics.metrics_history.append(metric)
            
        # Generate visualizations
        metrics_fig = self.visualizer._create_metrics_visualization()
        distribution_fig = self.visualizer._create_distribution_visualization()
        trends_fig = self.visualizer._create_trends_visualization('24H')
        
        self.assertIsNotNone(metrics_fig)
        self.assertIsNotNone(distribution_fig)
        self.assertIsNotNone(trends_fig)
        
    def test_alert_generation(self):
        """Test alert generation system"""
        # Add problematic metrics
        problem_metrics = self.test_metrics.copy()
        problem_metrics[-1]['cache_hit_rate'] = 0.3  # Should trigger alert
        
        for metric in problem_metrics:
            self.analytics.metrics_history.append(metric)
            
        # Start monitoring
        self.visualizer.start_monitoring()
        
        # Wait for alert generation
        time.sleep(self.visualizer.update_interval * 2)
        
        # Check if alert was generated
        self.assertFalse(self.visualizer.alert_queue.empty())
        alert = self.visualizer.alert_queue.get()
        self.assertEqual(alert['priority'], 'high')
        
    def test_analytics_storage(self):
        """Test analytics data storage and loading"""
        # Add test data
        self.analytics.metrics_history = self.test_metrics
        self.analytics.usage_patterns['test'] = self.test_usage_stats
        
        # Save analytics
        self.analytics.save_analytics()
        
        # Create new analytics instance and load data
        new_analytics = MemoryAnalytics(Path(self.temp_dir))
        new_analytics.load_analytics()
        
        # Verify data
        self.assertEqual(len(new_analytics.metrics_history), len(self.test_metrics))
        self.assertIn('test', new_analytics.usage_patterns)
        
    def test_trend_change_detection(self):
        """Test trend change detection"""
        # Add trend change to test data
        change_metrics = self.test_metrics.copy()
        for i in range(50, 100):
            change_metrics[i]['total_memories'] += i * 2  # Accelerating trend
            
        changes = self.trend_analyzer.analyze_trend_changes(
            change_metrics,
            'total_memories'
        )
        
        self.assertIsInstance(changes, list)
        self.assertGreater(len(changes), 0)
        self.assertIn('magnitude', changes[0])
        
    def test_metric_relationships(self):
        """Test metric relationship detection"""
        insights = self.trend_analyzer.analyze_trends(self.test_metrics)
        
        # Check for related metrics
        for metric, insight in insights.items():
            if insight.related_metrics:
                self.assertIsInstance(insight.related_metrics, list)
                self.assertGreater(len(insight.related_metrics), 0)
                
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main() 