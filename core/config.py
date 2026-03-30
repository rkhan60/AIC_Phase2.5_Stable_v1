"""Centralised configuration for all AIC system thresholds and parameters.

Import the singleton `config` object and read values from it rather than
scattering magic numbers throughout the codebase.  Override individual fields
at startup if you need environment-specific tuning.

Example::

    from core.config import config
    config.memory.importance_threshold = 0.6   # tighten retention
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MemoryConfig:
    """Memory system parameters."""
    importance_threshold: float = 0.50     # Min score to persist in long-term store
    half_life_days: int = 30               # Exponential decay half-life (days)
    working_memory_capacity: int = 100     # Max items kept in working memory
    consolidation_timeout_s: float = 1.0   # Queue.get() timeout in bg worker


@dataclass
class ReasoningConfig:
    """Reasoning quality thresholds."""
    min_step_confidence: float = 0.50      # Per-step confidence floor
    min_steps: int = 2                     # Minimum steps for a valid chain
    critique_score_threshold: float = 0.70 # Minimum critique score before retry
    max_retry_cycles: int = 3              # Max re-attempts per goal cycle


@dataclass
class GoalConfig:
    """Goal evaluation keyword lists and scoring boosts."""

    # --- Goal-type detection keywords ---
    immediate_keywords: List[str] = field(default_factory=lambda: [
        'fix', 'resolve', 'immediately', 'urgent', 'asap', 'now',
        'today', 'critical', 'emergency',
    ])
    tactical_keywords: List[str] = field(default_factory=lambda: [
        'improve', 'optimise', 'optimize', 'implement', 'build',
        'create', 'develop', 'launch', 'deploy', 'deliver',
    ])
    adaptive_keywords: List[str] = field(default_factory=lambda: [
        'adapt', 'respond', 'adjust', 'change', 'pivot', 'react',
    ])

    # --- Impact scoring keywords ---
    high_impact_keywords: List[str] = field(default_factory=lambda: [
        'revenue', 'profit', 'growth', 'efficiency', 'transform',
        'scale', 'optimise', 'optimize', 'market share', 'customer',
        'cost reduction', 'productivity', 'competitive',
    ])

    # --- Urgency scoring keywords ---
    urgency_keywords: List[str] = field(default_factory=lambda: [
        'urgent', 'immediately', 'asap', 'critical', 'deadline',
        'now', 'quickly', 'emergency', 'today',
    ])

    # --- Metric scoring weights ---
    resource_availability_boost: float = 0.20
    clear_criteria_boost: float = 0.15
    dependency_penalty_per_item: float = 0.10
    max_dependency_penalty: float = 0.30
    strategic_type_impact_boost: float = 0.15
    tactical_type_impact_boost: float = 0.05
    impact_keyword_boost_per_hit: float = 0.08
    max_impact_keyword_boost: float = 0.30
    urgency_keyword_boost: float = 0.35
    immediate_goal_type_boost: float = 0.35
    adaptive_goal_type_boost: float = 0.20
    time_constraint_boost: float = 0.15
    time_constraint_tight_boost: float = 0.30   # Used when constraint < 48 h
    alignment_context_boost: float = 0.10
    alignment_keyword_overlap_weight: float = 0.30


@dataclass
class CritiqueConfig:
    """Self-critic evaluation parameters."""
    ideal_trace_min_steps: int = 3
    ideal_trace_max_steps: int = 7
    conclusion_bonus: float = 0.15
    length_penalty_per_extra_step: float = 0.05


@dataclass
class AICConfig:
    """Top-level configuration container — import this."""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    critique: CritiqueConfig = field(default_factory=CritiqueConfig)


# ---------------------------------------------------------------------------
# Module-level singleton — import and override as needed
# ---------------------------------------------------------------------------
config = AICConfig()
