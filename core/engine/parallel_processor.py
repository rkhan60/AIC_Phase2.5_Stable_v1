"""parallel_processor — thread-based parallel agent execution (no PyTorch)."""

from typing import Dict, List, Optional
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from .business_agents import BusinessRole, BusinessAgent
from .enums import ConsultingRole  # noqa: F401


class ParallelAgentProcessor:
    """Manages parallel processing of agent tasks while maintaining collaboration."""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.task_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()
        self.agent_groups = [
            ['MARKET_ANALYST', 'FINANCIAL_STRATEGIST'],
            ['INNOVATION_STRATEGIST', 'DIGITAL_TRANSFORMATION_EXPERT'],
            ['RISK_MANAGER', 'SUSTAINABILITY_CONSULTANT'],
            ['CHANGE_MANAGEMENT_SPECIALIST', 'BUSINESS_DEVELOPMENT_EXPERT'],
        ]
        self.shared_context: Dict = {}

    def process_in_parallel(self, tasks: List[Dict],
                             agents: Dict[str, BusinessAgent]) -> List[Dict]:
        """Process tasks in parallel using agent groups."""
        results: List[Dict] = []
        grouped_tasks = self._group_tasks(tasks)

        with ThreadPoolExecutor(max_workers=len(self.agent_groups)) as executor:
            futures = []
            for group_idx, group in enumerate(self.agent_groups):
                if group_idx in grouped_tasks:
                    group_agents = [agents[role] for role in group if role in agents]
                    if not group_agents:
                        continue
                    future = executor.submit(
                        self._process_group_tasks,
                        grouped_tasks[group_idx],
                        group_agents,
                        group_idx,
                    )
                    futures.append(future)

            for future in as_completed(futures):
                group_results = future.result()
                results.extend(group_results)
                self._update_shared_context(group_results)

        return results

    def _group_tasks(self, tasks: List[Dict]) -> Dict[int, List[Dict]]:
        grouped: Dict[int, List[Dict]] = {}
        for task in tasks:
            idx = self._get_task_group(task)
            grouped.setdefault(idx, []).append(task)
            if len(grouped[idx]) >= self.batch_size:
                grouped[idx] = grouped[idx][: self.batch_size]
        return grouped

    def _get_task_group(self, task: Dict) -> int:
        type_to_group = {
            'market_analysis': 0,
            'innovation': 1,
            'risk_assessment': 2,
            'change_management': 3,
        }
        return type_to_group.get(task.get('type', ''), 0)

    def _process_group_tasks(self, tasks: List[Dict],
                              agents: List[BusinessAgent],
                              group_idx: int) -> List[Dict]:
        results = []
        context = self._get_relevant_context(group_idx)
        for task in tasks:
            task_input = task.get('input', task)
            intermediate = agents[0](task_input, context)
            final = agents[1](intermediate.get('output', intermediate), context) \
                if len(agents) > 1 else intermediate
            results.append({
                'task_id': task.get('id'),
                'intermediate_result': intermediate,
                'final_result': final,
            })
        return results

    def _update_shared_context(self, results: List[Dict]):
        for result in results:
            final = result.get('final_result', {})
            self.shared_context.update({
                'market_data': final.get('market_insights', {}),
                'innovation_data': final.get('innovation_insights', {}),
                'risk_data': final.get('risk_insights', {}),
                'change_data': final.get('change_insights', {}),
            })

    def _get_relevant_context(self, group_idx: int) -> Dict:
        keys = {0: ['market_data'], 1: ['innovation_data'],
                2: ['risk_data'], 3: ['change_data']}.get(group_idx, [])
        return {k: v for k, v in self.shared_context.items() if k in keys}


class AgentGroup:
    """A group of collaborating agents with shared context."""

    def __init__(self, agents: List[BusinessAgent],
                 shared_memory: Optional[Dict] = None):
        self.agents = agents
        self.shared_memory = shared_memory or {}
        self.group_context: Dict = {}

    def process_task(self, task: Dict,
                     context: Optional[Dict] = None) -> Dict:
        results = []
        current_output = task
        for agent in self.agents:
            combined = {**self.group_context, **(context or {}),
                        'previous_results': results}
            result = agent(current_output, combined)
            results.append(result)
            current_output = result.get('output', result)
            self._update_group_context(result)
        return {'task_id': task.get('id'), 'results': results,
                'final_output': current_output}

    def _update_group_context(self, result: Dict):
        if 'insights' in result:
            self.group_context.update(result['insights'])

    def share_insights(self, other_group: 'AgentGroup'):
        keys = {'market_insights', 'innovation_insights',
                'risk_insights', 'change_insights'}
        other_group.group_context.update(
            {k: v for k, v in self.group_context.items() if k in keys}
        )
