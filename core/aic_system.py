"""AIC Consulting Framework Engine.

This module provides the rule-based consulting intelligence layer used by the
AIC system.  It replaces the original file, which contained ~800 lines of
PyTorch neural-network classes that were never instantiated or imported
anywhere in the codebase.

What is kept / added:
  - BusinessIntelligenceType   — enum of BI analysis categories
  - ConsultingFramework        — enum of supported frameworks
  - BusinessContext            — dataclass capturing company context
  - ConsultingFrameworkEngine  — pure-Python engine that applies frameworks
                                 to a BusinessContext and returns structured
                                 analysis dicts ready for the pipeline
  - create_aic_system()        — thin shim so existing callers in
                                 main.py / diagnostics.py do not break
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BusinessIntelligenceType(Enum):
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    MARKET_DYNAMICS = "market_dynamics"
    VALUE_CHAIN_OPTIMIZATION = "value_chain_optimization"
    ORGANIZATIONAL_TRANSFORMATION = "organizational_transformation"
    FINANCIAL_PERFORMANCE = "financial_performance"
    STRATEGIC_POSITIONING = "strategic_positioning"


class ConsultingFramework(Enum):
    MECE_STRUCTURING = "mece_structuring"
    HYPOTHESIS_DRIVEN = "hypothesis_driven"
    PORTERS_FIVE_FORCES = "porters_five_forces"
    BCG_GROWTH_SHARE = "bcg_growth_share"
    ANSOFF_MATRIX = "ansoff_matrix"
    VALUE_CHAIN_ANALYSIS = "value_chain_analysis"
    MCKINSEY_7S = "mckinsey_7s"
    BAIN_RAPID = "bain_rapid"
    SWOT_ADVANCED = "swot_advanced"
    BLUE_OCEAN = "blue_ocean_strategy"


# ---------------------------------------------------------------------------
# Business context dataclass
# ---------------------------------------------------------------------------

@dataclass
class BusinessContext:
    """Comprehensive business context used as input to all framework engines."""
    industry: str = "general"
    company_size: str = "sme"                    # micro | sme | mid-market | enterprise
    market_position: str = "challenger"           # leader | challenger | follower | niche
    competitive_landscape: Dict = field(default_factory=dict)
    financial_health: Dict = field(default_factory=dict)
    organizational_maturity: str = "developing"   # nascent | developing | mature | optimised
    strategic_priorities: List[str] = field(default_factory=list)
    stakeholder_map: Dict = field(default_factory=dict)
    cultural_context: str = ""
    regulatory_environment: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Framework analysis templates
# ---------------------------------------------------------------------------

def _swot_template(ctx: BusinessContext, problem: str) -> Dict[str, Any]:
    return {
        "framework": ConsultingFramework.SWOT_ADVANCED.value,
        "problem": problem,
        "strengths": [
            f"Established presence in {ctx.industry} sector",
            f"Market position: {ctx.market_position}",
            f"Organisational maturity: {ctx.organizational_maturity}",
        ],
        "weaknesses": [
            "Areas requiring further investigation based on financial health data",
            f"Size constraints typical of {ctx.company_size} companies",
        ],
        "opportunities": [
            f"Strategic priorities identified: {', '.join(ctx.strategic_priorities) or 'TBD'}",
            "Market dynamics may present first-mover advantage",
        ],
        "threats": [
            f"Competitive landscape intensity: "
            f"{ctx.competitive_landscape.get('intensity', 'moderate')}",
            f"Regulatory considerations present: {bool(ctx.regulatory_environment)}",
        ],
        "recommendations": [
            "Leverage strengths to capture identified opportunities",
            "Develop mitigation plans for each material threat",
            "Address key weaknesses within 90 days",
        ],
    }


def _porters_five_forces_template(ctx: BusinessContext, problem: str) -> Dict[str, Any]:
    intensity = ctx.competitive_landscape.get("intensity", "moderate")
    return {
        "framework": ConsultingFramework.PORTERS_FIVE_FORCES.value,
        "problem": problem,
        "forces": {
            "competitive_rivalry": {
                "level": intensity,
                "drivers": ["Number of competitors", "Industry growth rate", "Switching costs"],
            },
            "threat_of_new_entrants": {
                "level": "medium",
                "drivers": ["Capital requirements", "Regulatory barriers", "Brand loyalty"],
            },
            "threat_of_substitutes": {
                "level": "medium",
                "drivers": ["Alternative technologies", "Price-performance trade-off"],
            },
            "buyer_power": {
                "level": "medium" if ctx.company_size in ("sme", "micro") else "low",
                "drivers": ["Buyer concentration", "Switching costs", "Price sensitivity"],
            },
            "supplier_power": {
                "level": "medium",
                "drivers": ["Supplier concentration", "Uniqueness of inputs"],
            },
        },
        "overall_attractiveness": (
            "low" if intensity == "high" else
            "high" if intensity == "low" else "medium"
        ),
        "recommendations": [
            "Reduce rivalry exposure through differentiation",
            "Build switching costs to lower buyer power",
            "Develop alternative supplier relationships",
        ],
    }


def _mckinsey_7s_template(ctx: BusinessContext, problem: str) -> Dict[str, Any]:
    return {
        "framework": ConsultingFramework.MCKINSEY_7S.value,
        "problem": problem,
        "elements": {
            "strategy": f"Address: {problem[:80]}",
            "structure": f"Aligned with {ctx.company_size} operating model",
            "systems": "Review current processes for efficiency gaps",
            "shared_values": ctx.cultural_context or "Define and communicate core values",
            "style": f"Leadership style appropriate for {ctx.organizational_maturity} maturity",
            "staff": "Ensure capability alignment with strategic priorities",
            "skills": f"Critical skills for {ctx.industry}: identify gaps vs. requirements",
        },
        "alignment_gaps": [
            "Strategy ↔ Structure alignment to be validated",
            "Systems ↔ Strategy consistency check recommended",
        ],
        "recommendations": [
            "Conduct 7S alignment workshop with leadership team",
            "Prioritise the two lowest-scoring elements for improvement",
            "Reassess alignment quarterly",
        ],
    }


def _mece_template(ctx: BusinessContext, problem: str) -> Dict[str, Any]:
    return {
        "framework": ConsultingFramework.MECE_STRUCTURING.value,
        "problem": problem,
        "issue_tree": {
            "root": problem,
            "branches": [
                {
                    "label": "Internal factors",
                    "leaves": ["Operational efficiency", "Financial performance", "People & capabilities"],
                },
                {
                    "label": "External factors",
                    "leaves": ["Market dynamics", "Competitive positioning", "Regulatory & macro environment"],
                },
            ],
        },
        "mutually_exclusive": True,
        "collectively_exhaustive": True,
        "next_steps": [
            "Quantify each branch with available data",
            "Identify the branch with highest impact × feasibility",
            "Develop hypotheses for top branch",
        ],
    }


def _ansoff_template(ctx: BusinessContext, problem: str) -> Dict[str, Any]:
    return {
        "framework": ConsultingFramework.ANSOFF_MATRIX.value,
        "problem": problem,
        "quadrants": {
            "market_penetration": {
                "description": "Grow share in existing markets with existing products",
                "risk": "low",
                "recommended": ctx.market_position in ("leader", "challenger"),
            },
            "market_development": {
                "description": "Enter new markets with existing products",
                "risk": "medium",
                "recommended": ctx.market_position == "leader",
            },
            "product_development": {
                "description": "New products for existing markets",
                "risk": "medium",
                "recommended": ctx.organizational_maturity in ("mature", "optimised"),
            },
            "diversification": {
                "description": "New products for new markets",
                "risk": "high",
                "recommended": False,
            },
        },
        "recommended_strategy": (
            "market_penetration"
            if ctx.market_position in ("challenger", "follower")
            else "product_development"
        ),
        "rationale": (
            f"Given {ctx.company_size} size and {ctx.market_position} position, "
            "lower-risk growth paths are preferred unless strong cash reserves exist."
        ),
        "recommendations": [
            "Validate recommended quadrant with financial capacity assessment",
            "Set 12-month OKRs aligned to chosen growth strategy",
        ],
    }


def _bcg_template(ctx: BusinessContext, problem: str) -> Dict[str, Any]:
    return {
        "framework": ConsultingFramework.BCG_GROWTH_SHARE.value,
        "problem": problem,
        "portfolio_guidance": {
            "stars": "High growth, high share — invest to maintain leadership",
            "cash_cows": "Low growth, high share — harvest to fund other units",
            "question_marks": "High growth, low share — selective investment needed",
            "dogs": "Low growth, low share — divest or turnaround",
        },
        "assessment_steps": [
            "Map all business units / product lines on the matrix",
            "Validate market growth rate and relative share data",
            "Allocate capital according to portfolio balance targets",
        ],
        "recommendations": [
            "Protect cash cow margins to fund star units",
            "Set explicit decision gates for each question mark",
        ],
    }


_FRAMEWORK_HANDLERS = {
    ConsultingFramework.SWOT_ADVANCED: _swot_template,
    ConsultingFramework.PORTERS_FIVE_FORCES: _porters_five_forces_template,
    ConsultingFramework.MCKINSEY_7S: _mckinsey_7s_template,
    ConsultingFramework.MECE_STRUCTURING: _mece_template,
    ConsultingFramework.ANSOFF_MATRIX: _ansoff_template,
    ConsultingFramework.BCG_GROWTH_SHARE: _bcg_template,
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ConsultingFrameworkEngine:
    """Pure-Python engine that applies consulting frameworks to a business problem.

    Usage::

        engine = ConsultingFrameworkEngine()
        context = BusinessContext(industry="retail", company_size="sme")
        result  = engine.analyse(
            problem="How do we improve customer retention?",
            context=context,
            frameworks=[ConsultingFramework.SWOT_ADVANCED,
                        ConsultingFramework.PORTERS_FIVE_FORCES],
        )
    """

    def analyse(
        self,
        problem: str,
        context: BusinessContext,
        frameworks: Optional[List[ConsultingFramework]] = None,
    ) -> Dict[str, Any]:
        """Run one or more consulting frameworks against a business problem.

        Args:
            problem:    Free-text problem statement.
            context:    BusinessContext describing the client company.
            frameworks: Frameworks to apply.  Defaults to
                        [SWOT_ADVANCED, PORTERS_FIVE_FORCES, MECE_STRUCTURING].

        Returns:
            Dict keyed by framework value, plus ``aggregated_recommendations``.
        """
        if not problem or not problem.strip():
            raise ValueError("problem statement must not be empty")

        if frameworks is None:
            frameworks = [
                ConsultingFramework.SWOT_ADVANCED,
                ConsultingFramework.PORTERS_FIVE_FORCES,
                ConsultingFramework.MECE_STRUCTURING,
            ]

        results: Dict[str, Any] = {
            "problem": problem,
            "context_summary": {
                "industry": context.industry,
                "size": context.company_size,
                "position": context.market_position,
            },
        }

        all_recommendations: List[str] = []
        for fw in frameworks:
            handler = _FRAMEWORK_HANDLERS.get(fw)
            if handler is None:
                logger.warning("No handler for framework %s — skipping", fw.value)
                continue
            try:
                fw_result = handler(context, problem)
                results[fw.value] = fw_result
                all_recommendations.extend(fw_result.get("recommendations", []))
            except Exception as exc:
                logger.error("Framework %s raised: %s", fw.value, exc)
                results[fw.value] = {"error": str(exc)}

        # Deduplicate while preserving order
        seen: set = set()
        unique_recs: List[str] = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)
        results["aggregated_recommendations"] = unique_recs
        return results

    def recommend_frameworks(self, context: BusinessContext) -> List[ConsultingFramework]:
        """Return the most relevant frameworks for the given business context."""
        suggestions: List[ConsultingFramework] = [ConsultingFramework.MECE_STRUCTURING]

        if context.competitive_landscape:
            suggestions.append(ConsultingFramework.PORTERS_FIVE_FORCES)

        if context.market_position in ("leader", "challenger"):
            suggestions.append(ConsultingFramework.BCG_GROWTH_SHARE)
            suggestions.append(ConsultingFramework.ANSOFF_MATRIX)

        if context.organizational_maturity in ("developing", "nascent"):
            suggestions.append(ConsultingFramework.MCKINSEY_7S)

        suggestions.append(ConsultingFramework.SWOT_ADVANCED)
        return suggestions


# ---------------------------------------------------------------------------
# Compatibility shim
# ---------------------------------------------------------------------------

def create_aic_system():
    """Delegate to core.engine.logic_engine.create_aic_system().

    Preserves backward compatibility for callers in main.py and diagnostics.py.
    """
    from core.engine.logic_engine import create_aic_system as _real
    return _real()
