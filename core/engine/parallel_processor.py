import torch
import torch.nn as nn
from typing import Dict, List, Optional
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from .business_agents import BusinessRole, BusinessAgent
from .enums import ConsultingRole

class ParallelAgentProcessor:
    """Manages parallel processing of agent tasks while maintaining collaboration"""
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.agent_groups = [
            ['MARKET_ANALYST', 'FINANCIAL_STRATEGIST'],
            ['INNOVATION_STRATEGIST', 'DIGITAL_TRANSFORMATION_EXPERT'],
            ['RISK_MANAGER', 'SUSTAINABILITY_CONSULTANT'],
            ['CHANGE_MANAGEMENT_SPECIALIST', 'BUSINESS_DEVELOPMENT_EXPERT']
        ]
        self.shared_context = {}
        
    def process_in_parallel(self, tasks: List[Dict], agents: Dict[str, BusinessAgent]) -> List[Dict]:
        """Process tasks in parallel using agent groups"""
        results = []
        
        # Group tasks by agent pairs
        grouped_tasks = self._group_tasks(tasks)
        
        with ThreadPoolExecutor(max_workers=len(self.agent_groups)) as executor:
            futures = []
            
            # Submit tasks for each agent group
            for group_idx, group in enumerate(self.agent_groups):
                if group_idx in grouped_tasks:
                    future = executor.submit(
                        self._process_group_tasks,
                        grouped_tasks[group_idx],
                        [agents[role] for role in group],
                        group_idx
                    )
                    futures.append(future)
            
            # Collect results while maintaining order
            for future in as_completed(futures):
                group_results = future.result()
                results.extend(group_results)
                
                # Update shared context for collaboration
                self._update_shared_context(group_results)
        
        return results
    
    def _group_tasks(self, tasks: List[Dict]) -> Dict[int, List[Dict]]:
        """Group tasks based on agent pairs"""
        grouped_tasks = {}
        
        for task in tasks:
            # Determine which group should handle the task
            group_idx = self._get_task_group(task)
            if group_idx not in grouped_tasks:
                grouped_tasks[group_idx] = []
            grouped_tasks[group_idx].append(task)
            
            # Split into batches if needed
            if len(grouped_tasks[group_idx]) >= self.batch_size:
                grouped_tasks[group_idx] = grouped_tasks[group_idx][:self.batch_size]
                
        return grouped_tasks
    
    def _get_task_group(self, task: Dict) -> int:
        """Determine which agent group should handle the task"""
        task_type = task.get('type', '')
        
        # Map task types to agent groups
        type_to_group = {
            'market_analysis': 0,      # MARKET_ANALYST + FINANCIAL_STRATEGIST
            'innovation': 1,           # INNOVATION_STRATEGIST + DIGITAL_TRANSFORMATION_EXPERT
            'risk_assessment': 2,      # RISK_MANAGER + SUSTAINABILITY_CONSULTANT
            'change_management': 3     # CHANGE_MANAGEMENT_SPECIALIST + BUSINESS_DEVELOPMENT_EXPERT
        }
        
        return type_to_group.get(task_type, 0)  # Default to first group if type unknown
    
    def _process_group_tasks(self, tasks: List[Dict], agents: List[BusinessAgent], group_idx: int) -> List[Dict]:
        """Process tasks using a specific agent group"""
        results = []
        
        for task in tasks:
            # Get relevant shared context
            context = self._get_relevant_context(group_idx)
            
            # Process with first agent
            intermediate_result = agents[0](task['input'], context)
            
            # Process with second agent, using first agent's output
            final_result = agents[1](intermediate_result['output'], context)
            
            results.append({
                'task_id': task.get('id'),
                'intermediate_result': intermediate_result,
                'final_result': final_result
            })
            
        return results
    
    def _update_shared_context(self, results: List[Dict]):
        """Update shared context with new insights"""
        for result in results:
            # Extract key insights
            insights = {
                'market_data': result['final_result'].get('market_insights', {}),
                'innovation_data': result['final_result'].get('innovation_insights', {}),
                'risk_data': result['final_result'].get('risk_insights', {}),
                'change_data': result['final_result'].get('change_insights', {})
            }
            
            # Update shared context
            self.shared_context.update(insights)
    
    def _get_relevant_context(self, group_idx: int) -> Dict:
        """Get context relevant to specific agent group"""
        context_mapping = {
            0: ['market_data'],              # Market + Financial
            1: ['innovation_data'],          # Innovation + Digital
            2: ['risk_data'],               # Risk + Sustainability
            3: ['change_data']              # Change + Business Development
        }
        
        relevant_keys = context_mapping.get(group_idx, [])
        return {k: v for k, v in self.shared_context.items() if k in relevant_keys}

class AgentGroup:
    """Represents a group of collaborating agents"""
    def __init__(self, agents: List[BusinessAgent], shared_memory: Dict = None):
        self.agents = agents
        self.shared_memory = shared_memory or {}
        self.group_context = {}
        
    def process_task(self, task: Dict, context: Optional[Dict] = None) -> Dict:
        """Process task using all agents in the group"""
        results = []
        current_output = task
        
        # Sequential processing through agents
        for agent in self.agents:
            # Combine task context with group context
            combined_context = {
                **self.group_context,
                **(context or {}),
                'previous_results': results
            }
            
            # Process with current agent
            result = agent(current_output, combined_context)
            results.append(result)
            current_output = result['output']
            
            # Update group context
            self._update_group_context(result)
        
        return {
            'task_id': task.get('id'),
            'results': results,
            'final_output': current_output
        }
    
    def _update_group_context(self, result: Dict):
        """Update group context with new insights"""
        if 'insights' in result:
            self.group_context.update(result['insights'])
            
    def share_insights(self, other_group: 'AgentGroup'):
        """Share relevant insights with another agent group"""
        shared_insights = {
            k: v for k, v in self.group_context.items()
            if k in ['market_insights', 'innovation_insights', 'risk_insights', 'change_insights']
        }
        other_group.group_context.update(shared_insights) 