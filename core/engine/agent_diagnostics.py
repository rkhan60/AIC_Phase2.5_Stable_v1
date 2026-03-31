"""agent_diagnostics.py — system diagnostics (no PyTorch dependency)."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .business_agents import BusinessRole, BusinessAgent, BusinessAgentManager
from .enums import ConsultingRole

logger = logging.getLogger(__name__)


@dataclass
class AgentDiagnostic:
    """Diagnostic results for a single agent class."""

    agent_name: str
    base_compliance: bool
    missing_methods: List[str]
    missing_capabilities: List[str]
    collaboration_support: bool
    async_support: bool
    memory_integration: bool
    business_summary: bool
    recommendations: List[str]


class AgentDiagnostics:
    """Inspect and report on agent health across the system."""

    _REQUIRED_METHODS = {
        "_execute_task_impl",
        "_create_business_summary",
        "collaborate",
        "add_insight",
        "add_artifact",
    }

    _REQUIRED_CAPABILITIES: Dict[str, set] = {
        "EDAVisualizer": {"DATA_ANALYSIS", "VISUALIZATION"},
        "CustomerModeler": {"CUSTOMER_MODELING", "PREDICTIVE_ANALYTICS"},
        "BIEngineer": {"BI_DEVELOPMENT", "DATA_PIPELINE"},
        "DataCleaner": {"DATA_CLEANING", "DATA_VALIDATION"},
        "LabelAgent": {"LABELING_SETUP", "ANNOTATION_MANAGEMENT"},
        "MentorAI": {"MENTORSHIP", "COURSE_CREATION", "COHORT_MANAGEMENT"},
    }

    def run_diagnostics(self) -> Dict[str, AgentDiagnostic]:
        """Attempt to import and diagnose known agent classes."""
        agent_classes: Dict[str, Any] = {}

        # Gracefully attempt imports; skip unavailable modules
        _try_imports = [
            ("EDAVisualizer", "core.agents.analytics_agents", "EDAVisualizer"),
            ("CustomerModeler", "core.agents.analytics_agents", "CustomerModeler"),
            ("BIEngineer", "core.agents.analytics_agents", "BIEngineer"),
            ("DataCleaner", "core.agents.data_operations_agents", "DataCleaner"),
            ("LabelAgent", "core.agents.data_operations_agents", "LabelAgent"),
            ("MentorAI", "core.agents.training_agents", "MentorAI"),
        ]
        for name, module_path, class_name in _try_imports:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                agent_classes[name] = getattr(mod, class_name)
            except Exception:
                pass

        diagnostics: Dict[str, AgentDiagnostic] = {}
        for name, cls in agent_classes.items():
            diagnostics[name] = self._diagnose_agent(name, cls)

        if not diagnostics:
            logger.warning("No agent classes found during diagnostics.")

        return diagnostics

    def _diagnose_agent(self, name: str, agent_class: Any) -> AgentDiagnostic:
        from .base_agent import BaseAgent  # local import to avoid hard dep at module level

        base_compliance = issubclass(agent_class, BaseAgent)
        methods = {m for m, _ in inspect.getmembers(agent_class, predicate=inspect.isfunction)}
        missing_methods = list(self._REQUIRED_METHODS - methods)

        required_caps = self._REQUIRED_CAPABILITIES.get(name, set())
        instance_caps: set = set()
        try:
            inst = agent_class.__new__(agent_class)
            if hasattr(inst, "capabilities"):
                instance_caps = {
                    c.name if hasattr(c, "name") else str(c)
                    for c in inst.capabilities
                }
        except Exception:
            pass
        missing_caps = list(required_caps - instance_caps)

        recs: List[str] = []
        if not base_compliance:
            recs.append(f"{name}: inherit from BaseAgent")
        for m in missing_methods:
            recs.append(f"{name}: implement {m}()")
        for c in missing_caps:
            recs.append(f"{name}: add capability {c}")

        return AgentDiagnostic(
            agent_name=name,
            base_compliance=base_compliance,
            missing_methods=missing_methods,
            missing_capabilities=missing_caps,
            collaboration_support="collaborate" in methods,
            async_support=any("async" in inspect.getsource(getattr(agent_class, m, lambda: None))
                              for m in methods if hasattr(agent_class, m)),
            memory_integration="memory" in str(inspect.getsource(agent_class)).lower(),
            business_summary="_create_business_summary" in methods,
            recommendations=recs,
        )

    def _measure_memory_usage(self, role: Any) -> float:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0
