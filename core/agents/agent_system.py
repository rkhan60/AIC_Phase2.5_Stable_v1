from typing import Dict, List, Any, Optional, Protocol
from datetime import datetime
from dataclasses import dataclass
import threading
import queue
from enum import Enum
import logging
from ..memory.memory_system import MemoryItem, MemoryType, EmotionalTag
from ..memory.symbolic_engine import SymbolicEngine
from ..memory.decay_system import DecayManager, MemoryOptimizer

class AgentType(Enum):
    MEMORY_MANAGER = "memory_manager"
    PATTERN_RECOGNIZER = "pattern_recognizer"
    EMOTIONAL_PROCESSOR = "emotional_processor"
    OPTIMIZATION_AGENT = "optimization_agent"
    CONSOLIDATION_AGENT = "consolidation_agent"

class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"

@dataclass
class AgentTask:
    """Task for agents to process"""
    task_id: str
    task_type: str
    priority: int
    data: Any
    created_at: datetime
    metadata: Dict[str, Any]

class AgentProtocol(Protocol):
    """Protocol defining agent interface"""
    def process_task(self, task: AgentTask) -> Any:
        pass

    def get_status(self) -> AgentStatus:
        pass

class BaseAgent:
    """Base class for all agents"""
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.status = AgentStatus.IDLE
        self.task_queue = queue.PriorityQueue()
        self.results: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True

    def start(self):
        """Start agent processing"""
        self.worker_thread.start()

    def stop(self):
        """Stop agent processing"""
        self._stop_event.set()
        self.worker_thread.join()

    def submit_task(self, task: AgentTask):
        """Submit task to agent"""
        self.task_queue.put((-task.priority, task))

    def _process_queue(self):
        """Process tasks from queue"""
        while not self._stop_event.is_set():
            try:
                if not self.task_queue.empty():
                    with self.lock:
                        self.status = AgentStatus.WORKING
                    _, task = self.task_queue.get()
                    result = self.process_task(task)
                    with self.lock:
                        self.results[task.task_id] = result
                        self.status = AgentStatus.IDLE
                    self.task_queue.task_done()
            except Exception as e:
                logging.error(f"Error in agent {self.agent_type}: {e}")
                with self.lock:
                    self.status = AgentStatus.ERROR

    def get_result(self, task_id: str) -> Optional[Any]:
        """Get task result"""
        with self.lock:
            return self.results.get(task_id)

class MemoryManagerAgent(BaseAgent):
    """Agent responsible for memory management"""
    def __init__(self, memory_system):
        super().__init__(AgentType.MEMORY_MANAGER)
        self.memory_system = memory_system

    def process_task(self, task: AgentTask) -> Any:
        if task.task_type == "add_memory":
            return self.memory_system.add_memory(**task.data)
        elif task.task_type == "retrieve_memory":
            return self.memory_system.retrieve_memory(**task.data)
        return None

class PatternRecognizerAgent(BaseAgent):
    """Agent for pattern recognition"""
    def __init__(self, symbolic_engine: SymbolicEngine):
        super().__init__(AgentType.PATTERN_RECOGNIZER)
        self.symbolic_engine = symbolic_engine

    def process_task(self, task: AgentTask) -> Any:
        if task.task_type == "find_patterns":
            return self.symbolic_engine.find_patterns(task.data)
        elif task.task_type == "process_memory":
            return self.symbolic_engine.process_memory(task.data)
        return None

class EmotionalProcessorAgent(BaseAgent):
    """Agent for emotional processing"""
    def __init__(self):
        super().__init__(AgentType.EMOTIONAL_PROCESSOR)
        self.emotional_weights = {
            EmotionalTag.URGENT: 1.0,
            EmotionalTag.IMPORTANT: 0.8,
            EmotionalTag.POSITIVE: 0.6,
            EmotionalTag.NEGATIVE: 0.7,
            EmotionalTag.NEUTRAL: 0.3
        }

    def process_task(self, task: AgentTask) -> Any:
        if task.task_type == "process_emotion":
            memory: MemoryItem = task.data
            total_weight = sum(self.emotional_weights[tag] for tag in memory.emotional_tags)
            return total_weight / len(memory.emotional_tags)
        return None

class OptimizationAgent(BaseAgent):
    """Agent for memory optimization"""
    def __init__(self, optimizer: MemoryOptimizer):
        super().__init__(AgentType.OPTIMIZATION_AGENT)
        self.optimizer = optimizer

    def process_task(self, task: AgentTask) -> Any:
        if task.task_type == "optimize_memory":
            memory, memory_id = task.data
            return self.optimizer.optimize_memory(memory, memory_id)
        return None

class ConsolidationAgent(BaseAgent):
    """Agent for memory consolidation"""
    def __init__(self, decay_manager: DecayManager):
        super().__init__(AgentType.CONSOLIDATION_AGENT)
        self.decay_manager = decay_manager

    def process_task(self, task: AgentTask) -> Any:
        if task.task_type == "consolidate_memory":
            memory, memory_id = task.data
            return self.decay_manager.process_memory(memory, memory_id)
        return None

class AgentSystem:
    """Coordinator for multi-agent system"""
    def __init__(self, memory_system, symbolic_engine: SymbolicEngine,
                 decay_manager: DecayManager, optimizer: MemoryOptimizer):
        self.agents: Dict[AgentType, BaseAgent] = {
            AgentType.MEMORY_MANAGER: MemoryManagerAgent(memory_system),
            AgentType.PATTERN_RECOGNIZER: PatternRecognizerAgent(symbolic_engine),
            AgentType.EMOTIONAL_PROCESSOR: EmotionalProcessorAgent(),
            AgentType.OPTIMIZATION_AGENT: OptimizationAgent(optimizer),
            AgentType.CONSOLIDATION_AGENT: ConsolidationAgent(decay_manager)
        }
        self.task_counter = 0
        self.lock = threading.Lock()

    def start_agents(self):
        """Start all agents"""
        for agent in self.agents.values():
            agent.start()

    def stop_agents(self):
        """Stop all agents"""
        for agent in self.agents.values():
            agent.stop()

    def submit_task(self, agent_type: AgentType, task_type: str,
                   data: Any, priority: int = 0,
                   metadata: Dict[str, Any] = None) -> str:
        """Submit task to specific agent"""
        with self.lock:
            self.task_counter += 1
            task_id = f"{agent_type.value}_{self.task_counter}"

        task = AgentTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data,
            created_at=datetime.now(),
            metadata=metadata or {}
        )

        self.agents[agent_type].submit_task(task)
        return task_id

    def get_result(self, agent_type: AgentType, task_id: str) -> Optional[Any]:
        """Get task result from agent"""
        return self.agents[agent_type].get_result(task_id)

    def get_agent_status(self, agent_type: AgentType) -> AgentStatus:
        """Get agent status"""
        return self.agents[agent_type].status

class AgentMonitor:
    """Monitor and analyze agent system performance"""
    def __init__(self, agent_system: AgentSystem):
        self.agent_system = agent_system
        self.performance_metrics: Dict[str, List[float]] = {}

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            agent_type: agent.status
            for agent_type, agent in self.agent_system.agents.items()
        }

    def get_queue_sizes(self) -> Dict[str, int]:
        """Get task queue sizes"""
        return {
            agent_type.value: agent.task_queue.qsize()
            for agent_type, agent in self.agent_system.agents.items()
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            'queue_sizes': self.get_queue_sizes(),
            'agent_status': self.get_system_status(),
            'metrics': self.performance_metrics
        } 