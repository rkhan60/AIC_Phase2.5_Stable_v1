"""agent_manager.py — central agent coordination and task routing."""

from __future__ import annotations

import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .business_agents import BusinessRole, BusinessAgent
from .enums import ConsultingRole, ReasoningType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentTask:
    task_id: str
    task_type: str
    input_data: Union[pd.DataFrame, Dict, str]
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AgentResult:
    task_id: str
    agent_id: str
    output_data: Union[pd.DataFrame, Dict]
    insights: Dict
    confidence: float
    processing_time: float
    metadata: Dict = field(default_factory=dict)


class AgentCapability(Enum):
    MARKET_ANALYSIS = "market_analysis"
    FINANCIAL_MODELING = "financial_modeling"
    RISK_ASSESSMENT = "risk_assessment"
    INNOVATION_EVALUATION = "innovation_evaluation"
    DIGITAL_TRANSFORMATION = "digital_transformation"
    SUSTAINABILITY_ASSESSMENT = "sustainability_assessment"
    CHANGE_MANAGEMENT = "change_management"
    BUSINESS_DEVELOPMENT = "business_development"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    CUSTOMER_MODELING = "customer_modeling"
    PREDICTIVE_ANALYTICS = "predictive_analytics"
    BI_DEVELOPMENT = "bi_development"
    DATA_PIPELINE = "data_pipeline"


class ModelType(Enum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"


# ---------------------------------------------------------------------------
# Agent profile
# ---------------------------------------------------------------------------

class AgentProfile:
    def __init__(
        self,
        agent_id: str,
        role: BusinessRole,
        capabilities: List[AgentCapability],
        models: Optional[Dict[ModelType, BaseEstimator]] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.models: Dict[ModelType, BaseEstimator] = models or {}
        self.performance_metrics = pd.DataFrame()
        self.task_history: List[str] = []

    def can_handle(self, task_type: str) -> bool:
        return any(cap.value == task_type for cap in self.capabilities)

    def update_performance(self, task_result: AgentResult) -> None:
        row = {
            "task_id": task_result.task_id,
            "processing_time": task_result.processing_time,
            "confidence": task_result.confidence,
            "timestamp": pd.Timestamp.now(),
        }
        self.performance_metrics = pd.concat(
            [self.performance_metrics, pd.DataFrame([row])],
            ignore_index=True,
        )
        self.task_history.append(task_result.task_id)


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------

class ModelManager:
    def __init__(self):
        self.models: Dict[str, BaseEstimator] = {}
        self.model_metadata: Dict[str, Dict] = {}

    def register_model(
        self,
        model_id: str,
        model: BaseEstimator,
        model_type: ModelType,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.models[model_id] = model
        self.model_metadata[model_id] = {
            "type": model_type,
            "metadata": metadata or {},
            "performance_metrics": {},
        }

    def get_model(self, model_id: str) -> Optional[BaseEstimator]:
        return self.models.get(model_id)

    def update_model_metrics(self, model_id: str, metrics: Dict) -> None:
        if model_id in self.model_metadata:
            self.model_metadata[model_id]["performance_metrics"].update(metrics)


# ---------------------------------------------------------------------------
# Insight pipeline
# ---------------------------------------------------------------------------

class InsightPipeline:
    def __init__(self):
        self.transformers: Dict[str, Callable] = {}
        self.insight_cache: Dict[str, Any] = {}

    def register_transformer(self, name: str, func: Callable) -> None:
        self.transformers[name] = func

    def process_prediction(
        self,
        prediction: Union[pd.DataFrame, np.ndarray],
        transformer_name: str,
        context: Optional[Dict] = None,
    ) -> Dict:
        if transformer_name not in self.transformers:
            return {"raw": str(prediction)}
        transformer = self.transformers[transformer_name]
        insight = transformer(prediction, context)
        cache_key = f"{transformer_name}_{hash(str(prediction))}"
        self.insight_cache[cache_key] = insight
        return insight


# ---------------------------------------------------------------------------
# Agent manager
# ---------------------------------------------------------------------------

class AgentManager:
    """Central manager for agent coordination and task routing."""

    def __init__(self):
        self.agents: Dict[str, AgentProfile] = {}
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.model_manager = ModelManager()
        self.insight_pipeline = InsightPipeline()

        self.routing_rules: Dict[str, List[BusinessRole]] = {
            AgentCapability.MARKET_ANALYSIS.value: [
                BusinessRole.MARKET_ANALYST,
                BusinessRole.FINANCIAL_STRATEGIST,
            ],
            AgentCapability.INNOVATION_EVALUATION.value: [
                BusinessRole.INNOVATION_STRATEGIST,
                BusinessRole.DIGITAL_TRANSFORMATION_EXPERT,
            ],
            AgentCapability.RISK_ASSESSMENT.value: [
                BusinessRole.RISK_MANAGER,
                BusinessRole.SUSTAINABILITY_CONSULTANT,
            ],
            AgentCapability.CHANGE_MANAGEMENT.value: [
                BusinessRole.CHANGE_MANAGEMENT_SPECIALIST,
                BusinessRole.BUSINESS_DEVELOPMENT_EXPERT,
            ],
        }

    def register_agent(self, agent: AgentProfile) -> None:
        self.agents[agent.agent_id] = agent
        logger.info("Registered agent %s (role=%s)", agent.agent_id, agent.role)

    def submit_task(self, task: AgentTask) -> None:
        self.task_queue.put((task.priority, task))
        logger.info("Submitted task %s (type=%s)", task.task_id, task.task_type)

    def process_tasks(self, max_workers: int = 4) -> List[AgentResult]:
        results: List[AgentResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            while not self.task_queue.empty():
                _, task = self.task_queue.get()
                for agent in self._find_suitable_agents(task.task_type):
                    futures.append(executor.submit(self._process_task, task, agent))
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _find_suitable_agents(self, task_type: str) -> List[AgentProfile]:
        roles = self.routing_rules.get(task_type, [])
        return [
            a for a in self.agents.values()
            if a.role in roles and a.can_handle(task_type)
        ]

    def _process_task(self, task: AgentTask, agent: AgentProfile) -> Optional[AgentResult]:
        try:
            _start = time.time()
            model = self._get_model_for_task(task, agent)
            prediction = model.predict(task.input_data)
            insights = self.insight_pipeline.process_prediction(
                prediction,
                f"{agent.role.value}_insights",
                {"task_type": task.task_type},
            )
            result = AgentResult(
                task_id=task.task_id,
                agent_id=agent.agent_id,
                output_data=prediction,
                insights=insights,
                confidence=self._calculate_confidence(prediction, model),
                processing_time=time.time() - _start,
                metadata={"model_id": model.__class__.__name__},
            )
            agent.update_performance(result)
            return result
        except Exception as exc:
            logger.error("Task %s failed for agent %s: %s", task.task_id, agent.agent_id, exc)
            return None

    def _get_model_for_task(self, task: AgentTask, agent: AgentProfile) -> BaseEstimator:
        """Return the best model for the task, matched by ModelType if specified."""
        requested_type = (task.metadata or {}).get("model_type")
        if requested_type:
            for model_type, model in agent.models.items():
                if model_type.value == requested_type:
                    return model
        return next(iter(agent.models.values()))

    def _calculate_confidence(
        self,
        prediction: Any,
        model: Optional[BaseEstimator] = None,
    ) -> float:
        try:
            if model is not None and hasattr(model, "predict_proba"):
                proba = model.predict_proba(
                    prediction if hasattr(prediction, "__len__") else [prediction]
                )
                return float(np.max(proba))
        except Exception:
            pass
        return 0.75


# ---------------------------------------------------------------------------
# Inter-agent messaging
# ---------------------------------------------------------------------------

class AgentMessaging:
    def __init__(self):
        self.message_queue: queue.Queue = queue.Queue()
        self.subscriptions: Dict[str, List[str]] = {}

    def send_message(self, from_agent: str, to_agent: str, message: Dict) -> None:
        self.message_queue.put({"from": from_agent, "to": to_agent, "content": message})

    def subscribe(self, agent_id: str, message_type: str) -> None:
        self.subscriptions.setdefault(message_type, []).append(agent_id)

    def get_messages(self, agent_id: str) -> List[Dict]:
        messages: List[Dict] = []
        tmp: queue.Queue = queue.Queue()
        while not self.message_queue.empty():
            msg = self.message_queue.get()
            if msg["to"] == agent_id:
                messages.append(msg)
            else:
                tmp.put(msg)
        while not tmp.empty():
            self.message_queue.put(tmp.get())
        return messages
