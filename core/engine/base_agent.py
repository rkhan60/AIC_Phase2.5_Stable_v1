from typing import Dict, List, Optional, Union, Any
import asyncio
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from ..memory.memory_manager import MemoryManager
from ..memory.symbolic_rule_engine import SymbolicRuleEngine
from ..memory.context_trigger import ContextTriggerSystem

logger = logging.getLogger(__name__)

@dataclass
class AgentAction:
    """Record of an agent's action"""
    action_id: str
    agent_id: str
    action_type: str
    timestamp: datetime
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    status: str = "pending"
    error: Optional[str] = None

@dataclass
class AgentMemory:
    """Agent's memory of past actions and context"""
    agent_id: str
    context: Dict[str, Any]
    action_history: List[AgentAction]
    insights: List[Dict[str, Any]]
    dependencies: List[str]
    artifacts: Dict[str, Any]

@dataclass
class BusinessSummary:
    """Business-facing summary of task execution"""
    key_points: List[str]
    recommendations: List[str]
    risks: List[str]
    next_steps: List[str]
    metrics: Dict[str, Any]
    value_delivered: float

class BaseAgent(ABC):
    """Base class for all AI consulting agents"""
    
    def __init__(self, agent_id: Optional[str] = None, memory_path: Optional[Path] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        
        # Initialize memory systems
        memory_path = memory_path or Path(f"agent_memory/{self.agent_id}")
        self.memory_manager = MemoryManager(memory_path)
        self.context_trigger = ContextTriggerSystem()
        
        # Load existing memory state
        self.memory_manager.load_state()
        
        self.project_tracker = None  # Set by AgentManager
        self.collaboration_queue = asyncio.Queue()
        self.status = "idle"
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task asynchronously"""
        try:
            # Record task start
            action = self._create_action("task_execution", task)
            self._record_action(action)
            
            # Add task to working memory
            self.memory_manager.working_memory.add_item(
                f"task_{action.action_id}",
                task,
                metadata={'action_id': action.action_id}
            )
            
            # Execute task
            self.status = "working"
            result = await self._execute_task_impl(task)
            
            # Generate business summary
            summary = self._create_business_summary(task, result)
            
            # Store task outcome in long-term memory
            self._store_task_outcome(task, result, summary)
            
            # Update action record
            action.status = "completed"
            action.result = {
                "technical_output": result,
                "business_summary": asdict(summary)
            }
            self._record_action(action)
            
            # Save memory state
            self.memory_manager.save_state()
            
            # Notify project tracker
            await self._update_project_tracker(action)
            
            return action.result
            
        except Exception as e:
            logger.error(f"Error in {self.agent_id}: {str(e)}")
            action.status = "failed"
            action.error = str(e)
            self._record_action(action)
            raise
        finally:
            self.status = "idle"
            
    async def collaborate(self, target_agent: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Request collaboration from another agent"""
        try:
            # Record collaboration request
            action = self._create_action("collaboration_request", {
                "target_agent": target_agent,
                "request": request
            })
            self._record_action(action)
            
            # Store collaboration context in working memory
            self.memory_manager.working_memory.add_item(
                f"collab_{action.action_id}",
                {
                    'target_agent': target_agent,
                    'request': request,
                    'status': 'pending'
                },
                metadata={'action_id': action.action_id}
            )
            
            # Add to collaboration queue
            await self.collaboration_queue.put({
                "source_agent": self.agent_id,
                "target_agent": target_agent,
                "request": request,
                "action": action
            })
            
            # Wait for response
            response = await self._wait_for_collaboration(action.action_id)
            
            # Update working memory with response
            self.memory_manager.working_memory.add_item(
                f"collab_response_{action.action_id}",
                response,
                metadata={'action_id': action.action_id}
            )
            
            # Update action record
            action.status = "completed"
            action.result = response
            self._record_action(action)
            
            return response
            
        except Exception as e:
            logger.error(f"Collaboration error in {self.agent_id}: {str(e)}")
            action.status = "failed"
            action.error = str(e)
            self._record_action(action)
            raise
            
    def update_context(self, context_update: Dict[str, Any]):
        """Update agent's context"""
        # Add to working memory
        self.memory_manager.working_memory.add_item(
            f"context_update_{datetime.now().isoformat()}",
            context_update
        )
        
        # Create context trigger
        self.context_trigger.add_structured_trigger(
            context_update,
            metadata={'timestamp': datetime.now().isoformat()}
        )
        
    def add_insight(self, insight: Dict[str, Any]):
        """Add new insight to agent's memory"""
        # Add to long-term memory
        self.memory_manager.long_term_memory.add_pattern(
            f"insight_{datetime.now().isoformat()}",
            insight
        )
        
        # Create semantic trigger
        if 'description' in insight:
            self.context_trigger.add_semantic_trigger(
                insight['description'],
                metadata={'insight_type': insight.get('type')}
            )
        
    def add_feedback(self, feedback: Dict[str, Any], sentiment: float):
        """Add feedback to emotional memory"""
        self.memory_manager.emotional_memory.add_feedback(
            f"feedback_{datetime.now().isoformat()}",
            feedback,
            sentiment
        )
        
    def find_relevant_memories(self, context: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Find memories relevant to given context"""
        # Find matching triggers
        matching_triggers = self.context_trigger.find_matching_triggers(context)
        
        relevant_memories = {
            'patterns': [],
            'feedback': [],
            'working_context': []
        }
        
        for trigger in matching_triggers:
            if trigger.trigger_type == 'structured':
                # Find relevant patterns
                patterns = self.memory_manager.long_term_memory.find_relevant_patterns(trigger.content)
                relevant_memories['patterns'].extend(patterns)
                
            elif trigger.trigger_type == 'semantic':
                # Get sentiment summary for similar contexts
                sentiment = self.memory_manager.emotional_memory.get_sentiment_summary(
                    {'context_similarity': trigger.content}
                )
                relevant_memories['feedback'].append(sentiment)
        
        return relevant_memories
        
    @abstractmethod
    async def _execute_task_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of task execution logic"""
        pass
        
    @abstractmethod
    def _create_business_summary(self, task: Dict[str, Any], result: Dict[str, Any]) -> BusinessSummary:
        """Create business-facing summary of task execution"""
        pass
        
    def _create_action(self, action_type: str, parameters: Dict[str, Any]) -> AgentAction:
        """Create new action record"""
        return AgentAction(
            action_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            action_type=action_type,
            timestamp=datetime.now(),
            parameters=parameters
        )
        
    def _record_action(self, action: AgentAction):
        """Record action in agent's memory"""
        self.memory_manager.working_memory.add_item(
            f"action_{action.action_id}",
            asdict(action)
        )
        
    def _store_task_outcome(self, task: Dict[str, Any], result: Dict[str, Any], summary: BusinessSummary):
        """Store task outcome in long-term memory"""
        # Create pattern from task outcome
        pattern = {
            'conditions': {
                'task_type': task.get('type'),
                'input_parameters': task.get('parameters', {})
            },
            'consequences': {
                'approach': result.get('approach'),
                'outcome': result.get('outcome'),
                'success_metrics': summary.metrics
            },
            'confidence': summary.value_delivered
        }
        
        self.memory_manager.long_term_memory.add_pattern(
            f"task_pattern_{datetime.now().isoformat()}",
            pattern,
            metadata={'task_id': task.get('id')}
        )
        
    async def _update_project_tracker(self, action: AgentAction):
        """Update project tracker with action results"""
        if self.project_tracker:
            await self.project_tracker.record_agent_action(action)
            
    async def _wait_for_collaboration(self, action_id: str, timeout: float = 300) -> Dict[str, Any]:
        """Wait for collaboration response"""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if action := self.memory_manager.working_memory.get_item(f"action_{action_id}"):
                if action['status'] == "completed":
                    return action['result']
                elif action['status'] == "failed":
                    raise Exception(f"Collaboration failed: {action['error']}")
            await asyncio.sleep(1)
        raise TimeoutError("Collaboration request timed out") 