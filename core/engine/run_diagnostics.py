import asyncio
import logging
from pathlib import Path
from typing import Dict, List

from .agent_diagnostics import AgentDiagnostics, AgentUpdater
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

async def run_agent_diagnostics():
    """Run diagnostics on all agents and generate update recommendations"""
    
    # Initialize diagnostics
    diagnostics = AgentDiagnostics()
    
    # Run diagnostics
    results = diagnostics.run_diagnostics()
    
    # Print diagnostic results
    print("\n=== Agent Diagnostics Results ===\n")
    for agent_name, diagnostic in results.items():
        print(f"\nAgent: {agent_name}")
        print(f"Base Compliance: {'✓' if diagnostic.base_compliance else '✗'}")
        print(f"Missing Methods: {', '.join(diagnostic.missing_methods) if diagnostic.missing_methods else 'None'}")
        print(f"Missing Capabilities: {', '.join(diagnostic.missing_capabilities) if diagnostic.missing_capabilities else 'None'}")
        print(f"Collaboration Support: {'✓' if diagnostic.collaboration_support else '✗'}")
        print(f"Async Support: {'✓' if diagnostic.async_support else '✗'}")
        print(f"Memory Integration: {'✓' if diagnostic.memory_integration else '✗'}")
        print(f"Business Summary: {'✓' if diagnostic.business_summary else '✗'}")
        
        if diagnostic.recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(diagnostic.recommendations, 1):
                print(f"{i}. {rec}")
    
    # Generate updates
    updater = AgentUpdater(results)
    updates = updater.generate_updates()
    
    # Print update plans
    print("\n=== Agent Update Plans ===\n")
    for agent_name, update_plan in updates.items():
        if update_plan:
            print(f"\nUpdates needed for {agent_name}:")
            for i, update in enumerate(update_plan, 1):
                print(f"\n{i}. Update:")
                print(update)
        else:
            print(f"\n{agent_name} is up to date!")
            
    return results, updates

if __name__ == "__main__":
    asyncio.run(run_agent_diagnostics()) 