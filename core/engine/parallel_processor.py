"""parallel_processor.py — thread-based parallel agent processing (no PyTorch)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .business_agents import BusinessRole, BusinessAgent


class AgentGroup:
    """A pair of agents that collaborate on related tasks."""

    def __init__(self, agents: List[BusinessAgent]):
        self.agents = agents

    def process_task(
        self,
        task: Dict[str, Any],
        shared_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        current_output: Dict[str, Any] = dict(task)
        for agent in self.agents:
            current_output = agent(current_output, shared_context)
        return current_output


class ParallelAgentProcessor:
    """Runs AgentGroups concurrently via a ThreadPoolExecutor."""

    _GROUP_ROLES: List[List[BusinessRole]] = [
        [BusinessRole.MARKET_ANALYST, BusinessRole.FINANCIAL_STRATEGIST],
        [BusinessRole.INNOVATION_STRATEGIST, BusinessRole.DIGITAL_TRANSFORMATION_EXPERT],
        [BusinessRole.RISK_MANAGER, BusinessRole.SUSTAINABILITY_CONSULTANT],
        [BusinessRole.CHANGE_MANAGEMENT_SPECIALIST, BusinessRole.BUSINESS_DEVELOPMENT_EXPERT],
    ]

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.shared_context: Dict[str, Any] = {}

    def process_in_parallel(
        self,
        tasks: List[Dict],
        agents: Dict[str, BusinessAgent],
    ) -> List[Dict]:
        groups = [
            AgentGroup([agents.get(r.value, BusinessAgent(r)) for r in role_list])
            for role_list in self._GROUP_ROLES
        ]

        results: List[Dict] = []
        bounded_tasks = tasks[: self.batch_size]

        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(group.process_task, task, self.shared_context): task
                for task in bounded_tasks
                for group in groups
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                self.shared_context.update(result.get("context_update", {}))

        return results
