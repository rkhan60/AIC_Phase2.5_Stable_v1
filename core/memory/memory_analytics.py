from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

@dataclass
class MemoryMetrics:
    """Metrics for memory system performance"""
    total_memories: int
    active_memories: int  # Accessed in last 24 hours
    consolidated_count: int
    abstract_rules_count: int
    avg_confidence: float
    avg_relevance: float
    cache_hit_rate: float
    query_latency_ms: float

@dataclass
class MemoryUsageStats:
    """Statistics about memory usage patterns"""
    access_frequency: Dict[str, int]  # Memory ID -> access count
    context_distribution: Dict[str, int]  # Context type -> usage count
    confidence_distribution: Dict[str, float]  # Range -> percentage
    memory_age_distribution: Dict[str, int]  # Age range -> count
    top_accessed_patterns: List[str]  # Most frequently accessed patterns

class MemoryAnalytics:
    """Analytics system for memory management"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("memory_analytics")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[Dict[str, Any]] = []
        self.usage_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_logs: List[Dict[str, Any]] = []
        self.query_logs: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.query_start_times: Dict[str, datetime] = {}
        self.cache_access_counts = {'hits': 0, 'misses': 0}
        
    def start_query_tracking(self, query_id: str):
        """Start tracking a memory query"""
        self.query_start_times[query_id] = datetime.now()
        
    def end_query_tracking(self, query_id: str, query_data: Dict[str, Any]):
        """End tracking a memory query and log results"""
        if query_id in self.query_start_times:
            duration = datetime.now() - self.query_start_times[query_id]
            
            log_entry = {
                'query_id': query_id,
                'timestamp': datetime.now().isoformat(),
                'duration_ms': duration.total_seconds() * 1000,
                'query_type': query_data.get('type', 'unknown'),
                'results_count': query_data.get('results_count', 0),
                'confidence_threshold': query_data.get('confidence_threshold', 0),
                'success': query_data.get('success', False)
            }
            
            self.query_logs.append(log_entry)
            del self.query_start_times[query_id]
            
    def track_cache_access(self, hit: bool):
        """Track cache hit/miss"""
        if hit:
            self.cache_access_counts['hits'] += 1
        else:
            self.cache_access_counts['misses'] += 1
            
    def collect_metrics(self, memory_manager, memory_optimizer) -> MemoryMetrics:
        """Collect current memory system metrics"""
        # Calculate basic metrics
        total_memories = len(memory_manager.long_term_memory.items)
        active_memories = len(memory_optimizer._identify_hot_memories())
        consolidated_count = len([m for m in memory_manager.long_term_memory.items.values()
                                if m.metadata.get('merged_count', 0) > 0])
        abstract_rules = len([r for r in memory_manager.rule_engine.rules
                            if r.project_context.get('type') == 'abstract_pattern'])
        
        # Calculate averages
        confidences = [m.confidence for m in memory_manager.long_term_memory.items.values()]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Get recent query performance
        recent_queries = self.query_logs[-100:] if self.query_logs else []
        avg_relevance = np.mean([q.get('relevance_score', 0) for q in recent_queries]) if recent_queries else 0.0
        avg_latency = np.mean([q.get('duration_ms', 0) for q in recent_queries]) if recent_queries else 0.0
        
        # Calculate cache hit rate
        total_cache_accesses = sum(self.cache_access_counts.values())
        cache_hit_rate = (self.cache_access_counts['hits'] / total_cache_accesses 
                         if total_cache_accesses > 0 else 0.0)
        
        metrics = MemoryMetrics(
            total_memories=total_memories,
            active_memories=active_memories,
            consolidated_count=consolidated_count,
            abstract_rules_count=abstract_rules,
            avg_confidence=avg_confidence,
            avg_relevance=avg_relevance,
            cache_hit_rate=cache_hit_rate,
            query_latency_ms=avg_latency
        )
        
        # Store metrics history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            **metrics.__dict__
        })
        
        return metrics
        
    def analyze_usage_patterns(self, memory_manager, memory_optimizer) -> MemoryUsageStats:
        """Analyze memory usage patterns"""
        # Analyze access frequency
        access_frequency = defaultdict(int)
        for memory_id, accesses in memory_optimizer.access_patterns.items():
            access_frequency[memory_id] = len(accesses)
            
        # Analyze context distribution
        context_distribution = defaultdict(int)
        for item in memory_manager.long_term_memory.items.values():
            for context_type in item.metadata.get('context_types', []):
                context_distribution[context_type] += 1
                
        # Analyze confidence distribution
        confidences = [m.confidence for m in memory_manager.long_term_memory.items.values()]
        confidence_ranges = {
            'low': len([c for c in confidences if c < 0.4]),
            'medium': len([c for c in confidences if 0.4 <= c < 0.7]),
            'high': len([c for c in confidences if c >= 0.7])
        }
        
        # Analyze memory age distribution
        now = datetime.now()
        age_distribution = defaultdict(int)
        for item in memory_manager.long_term_memory.items.values():
            age_days = (now - item.created_at).days
            if age_days < 7:
                age_distribution['1_week'] += 1
            elif age_days < 30:
                age_distribution['1_month'] += 1
            else:
                age_distribution['older'] += 1
                
        # Get top accessed patterns
        sorted_patterns = sorted(
            access_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_patterns = [p[0] for p in sorted_patterns[:10]]
        
        stats = MemoryUsageStats(
            access_frequency=dict(access_frequency),
            context_distribution=dict(context_distribution),
            confidence_distribution=confidence_ranges,
            memory_age_distribution=dict(age_distribution),
            top_accessed_patterns=top_patterns
        )
        
        # Store usage patterns
        self.usage_patterns['access_patterns'].append({
            'timestamp': datetime.now().isoformat(),
            **stats.__dict__
        })
        
        return stats
        
    def generate_optimization_recommendations(self, 
                                           metrics: MemoryMetrics,
                                           usage_stats: MemoryUsageStats) -> List[Dict[str, Any]]:
        """Generate recommendations for memory system optimization"""
        recommendations = []
        
        # Check cache performance
        if metrics.cache_hit_rate < 0.7:
            recommendations.append({
                'type': 'cache_optimization',
                'priority': 'high',
                'description': 'Cache hit rate is below target. Consider increasing cache size or adjusting caching strategy.',
                'metrics': {'current_hit_rate': metrics.cache_hit_rate, 'target': 0.7}
            })
            
        # Check memory consolidation
        consolidation_ratio = metrics.consolidated_count / metrics.total_memories
        if consolidation_ratio < 0.2:
            recommendations.append({
                'type': 'consolidation',
                'priority': 'medium',
                'description': 'Low memory consolidation ratio. Consider adjusting consolidation thresholds.',
                'metrics': {'current_ratio': consolidation_ratio, 'target': 0.2}
            })
            
        # Check query performance
        if metrics.query_latency_ms > 100:  # 100ms threshold
            recommendations.append({
                'type': 'query_optimization',
                'priority': 'high',
                'description': 'Query latency exceeds target. Consider optimization or scaling.',
                'metrics': {'current_latency': metrics.query_latency_ms, 'target': 100}
            })
            
        # Check memory distribution
        if len(usage_stats.confidence_distribution.get('low', [])) > metrics.total_memories * 0.3:
            recommendations.append({
                'type': 'confidence_improvement',
                'priority': 'medium',
                'description': 'High proportion of low-confidence memories. Review confidence calculation.',
                'metrics': {'low_confidence_ratio': len(usage_stats.confidence_distribution['low']) / metrics.total_memories}
            })
            
        return recommendations
        
    def save_analytics(self):
        """Save analytics data to storage"""
        analytics_data = {
            'metrics_history': self.metrics_history,
            'usage_patterns': dict(self.usage_patterns),
            'performance_logs': self.performance_logs,
            'query_logs': self.query_logs
        }
        
        with open(self.storage_path / 'memory_analytics.json', 'w') as f:
            json.dump(analytics_data, f, indent=2)
            
    def load_analytics(self):
        """Load analytics data from storage"""
        if (self.storage_path / 'memory_analytics.json').exists():
            with open(self.storage_path / 'memory_analytics.json') as f:
                analytics_data = json.load(f)
                
            self.metrics_history = analytics_data['metrics_history']
            self.usage_patterns = defaultdict(list, analytics_data['usage_patterns'])
            self.performance_logs = analytics_data['performance_logs']
            self.query_logs = analytics_data['query_logs']
            
    def generate_analytics_report(self, time_period: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        now = datetime.now()
        start_time = now - time_period
        
        # Filter recent data
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > start_time
        ]
        
        recent_queries = [
            q for q in self.query_logs
            if datetime.fromisoformat(q['timestamp']) > start_time
        ]
        
        # Calculate trends
        metrics_trends = self._calculate_trends(recent_metrics)
        performance_trends = self._calculate_query_performance_trends(recent_queries)
        
        # Generate report
        report = {
            'time_period': {
                'start': start_time.isoformat(),
                'end': now.isoformat()
            },
            'summary_metrics': {
                'total_queries': len(recent_queries),
                'avg_latency': np.mean([q['duration_ms'] for q in recent_queries]) if recent_queries else 0,
                'success_rate': len([q for q in recent_queries if q['success']]) / len(recent_queries) if recent_queries else 0
            },
            'metrics_trends': metrics_trends,
            'performance_trends': performance_trends,
            'recommendations': self.generate_optimization_recommendations(
                MemoryMetrics(**recent_metrics[-1]) if recent_metrics else None,
                self.analyze_usage_patterns(None, None)  # Pass actual instances in real usage
            )
        }
        
        return report
        
    def _calculate_trends(self, metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trends in metrics over time"""
        if not metrics or len(metrics) < 2:
            return {}
            
        trends = {}
        first, last = metrics[0], metrics[-1]
        
        for key in ['total_memories', 'active_memories', 'avg_confidence', 'cache_hit_rate']:
            if key in first and key in last:
                change = (last[key] - first[key]) / first[key] if first[key] != 0 else 0
                trends[key] = change
                
        return trends
        
    def _calculate_query_performance_trends(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trends in query performance"""
        if not queries:
            return {}
            
        # Group queries by type
        queries_by_type = defaultdict(list)
        for query in queries:
            queries_by_type[query['query_type']].append(query)
            
        trends = {}
        for query_type, type_queries in queries_by_type.items():
            # Calculate average latency trend
            latencies = [q['duration_ms'] for q in type_queries]
            if len(latencies) >= 2:
                latency_change = (latencies[-1] - latencies[0]) / latencies[0] if latencies[0] != 0 else 0
                trends[f'{query_type}_latency_trend'] = latency_change
                
            # Calculate success rate trend
            success_rates = [
                len([q for q in type_queries[:i+1] if q['success']]) / (i+1)
                for i in range(len(type_queries))
            ]
            if len(success_rates) >= 2:
                success_trend = success_rates[-1] - success_rates[0]
                trends[f'{query_type}_success_trend'] = success_trend
                
        return trends 