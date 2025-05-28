from core.engine import (
    create_aic_system,
    AgentDiagnostics
)
import json
from pathlib import Path

def run_diagnostics():
    """Run comprehensive diagnostics on the AIC system"""
    print("\n🔍 Running AIC System Diagnostics")
    print("="*50)
    
    # Create AIC system
    print("\nInitializing AIC system...")
    model, _, _ = create_aic_system()
    
    # Initialize diagnostics
    diagnostics = AgentDiagnostics(model)
    
    # Run analysis
    print("\nAnalyzing agent efficiency...")
    report = diagnostics.generate_optimization_report()
    
    # Save report
    output_dir = Path(__file__).parent / "diagnostics"
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "agent_optimization_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display results
    print("\n📊 Current System Metrics:")
    print("-" * 30)
    efficiency = report['current_metrics']['system_efficiency']
    print(f"Average Response Time: {efficiency['avg_response_time']:.3f}s")
    print(f"Average Memory Usage: {efficiency['avg_memory_usage']:.1f}MB")
    print(f"Average Utilization: {efficiency['avg_utilization']*100:.1f}%")
    
    print("\n⚡ Optimization Suggestions:")
    print("-" * 30)
    
    # Parallel Processing
    parallel = report['optimization_suggestions']['parallel_processing']
    print(f"\n1. Parallel Processing (Expected Speedup: {parallel['estimated_speedup']})")
    print("   Recommended agent groups:")
    for group in parallel['parallel_agent_groups']:
        print(f"   - {' + '.join(group)}")
    
    # Memory Sharing
    memory = report['optimization_suggestions']['memory_sharing']
    print(f"\n2. Memory Optimization (Expected Reduction: {memory['estimated_memory_reduction']})")
    print("   Shared memory pools:")
    for pool in memory['shared_memory_pools']:
        print(f"   - {pool['name']}: {', '.join(pool['agents'])}")
    
    # Agent Grouping
    grouping = report['optimization_suggestions']['agent_grouping']
    print(f"\n3. Agent Grouping (Expected Efficiency Gain: {grouping['estimated_efficiency_gain']})")
    for group_name, agents in grouping.items():
        if group_name != 'estimated_efficiency_gain':
            print(f"   - {group_name.replace('_', ' ').title()}: {', '.join(agents)}")
    
    print("\n🎯 Recommended Actions:")
    print("-" * 30)
    print("\nImmediate Actions:")
    for action in report['recommendations']['immediate_actions']:
        print(f"✓ {action}")
    
    print("\nLong-term Improvements:")
    for improvement in report['recommendations']['long_term_improvements']:
        print(f"• {improvement}")
    
    print(f"\n✨ Full report saved to: {report_file}")

if __name__ == "__main__":
    run_diagnostics() 