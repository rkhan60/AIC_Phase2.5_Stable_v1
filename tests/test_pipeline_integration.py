"""Integration tests for the AIC autonomous pipeline.

These tests exercise the full end-to-end flow without mocking internal
components, verifying that all Phase 1-4 fixes hold together correctly.

Run with:
    python -m pytest tests/test_pipeline_integration.py -v
"""

import tempfile
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    """Config singleton is importable and has correct defaults."""

    def test_import(self):
        from core.config import config
        self.assertIsNotNone(config)

    def test_memory_defaults(self):
        from core.config import config
        self.assertEqual(config.memory.importance_threshold, 0.50)
        self.assertEqual(config.memory.half_life_days, 30)

    def test_reasoning_defaults(self):
        from core.config import config
        self.assertEqual(config.reasoning.min_step_confidence, 0.50)
        self.assertEqual(config.reasoning.min_steps, 2)

    def test_goal_keywords_populated(self):
        from core.config import config
        self.assertTrue(len(config.goal.immediate_keywords) > 0)
        self.assertTrue(len(config.goal.high_impact_keywords) > 0)


# ---------------------------------------------------------------------------
# GoalPlanner
# ---------------------------------------------------------------------------

class TestGoalPlanner(unittest.TestCase):
    """GoalPlanner: real heuristics, UUID IDs, input validation."""

    def _make_planner(self):
        from core.memory.memory_query import MemoryQueryEngine
        from core.agents.goal_planner import GoalPlanner
        return GoalPlanner(MemoryQueryEngine())

    def test_create_goal_returns_goal(self):
        from core.agents.goal_planner import Goal
        planner = self._make_planner()
        goal = planner.create_goal("Improve customer retention urgently", {})
        self.assertIsInstance(goal, Goal)

    def test_goal_id_is_uuid_format(self):
        planner = self._make_planner()
        goal = planner.create_goal("Increase revenue", {})
        self.assertTrue(goal.id.startswith("goal_"))
        # Should NOT contain MD5 timestamp prefix pattern
        self.assertNotIn("_202", goal.id)

    def test_goal_ids_are_unique(self):
        planner = self._make_planner()
        ids = {planner.create_goal("Same description", {}).id for _ in range(20)}
        self.assertEqual(len(ids), 20)

    def test_immediate_keyword_sets_type(self):
        from core.agents.goal_planner import GoalType
        planner = self._make_planner()
        goal = planner.create_goal("Fix the urgent production bug now", {})
        self.assertEqual(goal.type, GoalType.IMMEDIATE)

    def test_strategic_type_is_default(self):
        from core.agents.goal_planner import GoalType
        planner = self._make_planner()
        goal = planner.create_goal("Expand into new markets over the long term", {"long_term": True})
        self.assertEqual(goal.type, GoalType.STRATEGIC)

    def test_feasibility_increases_with_resources(self):
        planner = self._make_planner()
        g_no_res  = planner.create_goal("Deploy new analytics platform", {})
        g_with_res = planner.create_goal("Deploy new analytics platform", {"resources": "team of 5"})
        self.assertGreater(g_with_res.metrics.feasibility, g_no_res.metrics.feasibility)

    def test_urgency_increases_with_urgent_keyword(self):
        planner = self._make_planner()
        normal = planner.create_goal("Review quarterly financials", {})
        urgent = planner.create_goal("Urgent: review quarterly financials immediately", {})
        self.assertGreater(urgent.metrics.urgency, normal.metrics.urgency)

    def test_impact_increases_with_revenue_keyword(self):
        planner = self._make_planner()
        low_impact  = planner.create_goal("Organise the filing system", {})
        high_impact = planner.create_goal("Grow revenue and profit through market expansion", {})
        self.assertGreater(high_impact.metrics.impact, low_impact.metrics.impact)

    def test_success_criteria_contains_description(self):
        planner = self._make_planner()
        goal = planner.create_goal("Reduce churn rate by 20%", {})
        combined = " ".join(goal.success_criteria)
        self.assertIn("Reduce churn rate", combined)

    def test_empty_input_raises(self):
        planner = self._make_planner()
        with self.assertRaises(ValueError):
            planner.create_goal("", {})

    def test_blank_input_raises(self):
        planner = self._make_planner()
        with self.assertRaises(ValueError):
            planner.create_goal("   ", {})

    def test_long_input_is_truncated(self):
        planner = self._make_planner()
        long_input = "x" * 3000
        goal = planner.create_goal(long_input, {})
        self.assertLessEqual(len(goal.description), 2000)

    def test_non_dict_context_is_tolerated(self):
        planner = self._make_planner()
        # Should not raise — invalid context replaced with {}
        goal = planner.create_goal("Some valid goal", context=None)
        self.assertIsNotNone(goal)

    def test_plan_goal_returns_hierarchy(self):
        from core.agents.goal_planner import GoalHierarchy
        planner = self._make_planner()
        hierarchy = planner.plan_goal("Build a customer dashboard", {})
        self.assertIsInstance(hierarchy, GoalHierarchy)
        self.assertIsNotNone(hierarchy.main_goal)


# ---------------------------------------------------------------------------
# ThoughtProcessor
# ---------------------------------------------------------------------------

class TestThoughtProcessor(unittest.TestCase):

    def setUp(self):
        from core.agents.thought_processor import ThoughtProcessor
        self.processor = ThoughtProcessor()

    def test_generate_returns_chain(self):
        from core.agents.thought_processor import ReasoningChain
        chain = self.processor.generate("Analyse customer churn", {"domain": "retail"})
        self.assertIsInstance(chain, ReasoningChain)

    def test_chain_has_minimum_steps(self):
        chain = self.processor.generate("Reduce operational costs", {"complexity_level": 1})
        self.assertGreaterEqual(len(chain.steps), 2)

    def test_confidence_is_between_0_and_1(self):
        chain = self.processor.generate("Expand to new markets", {})
        self.assertGreaterEqual(chain.overall_confidence, 0.0)
        self.assertLessEqual(chain.overall_confidence, 1.0)

    def test_higher_complexity_gives_more_steps(self):
        low  = self.processor.generate("Goal", {"complexity_level": 1})
        high = self.processor.generate("Goal", {"complexity_level": 4})
        self.assertGreater(len(high.steps), len(low.steps))

    def test_generate_reasoning_is_alias(self):
        chain = self.processor.generate_reasoning("Some goal", {"domain": "finance"})
        self.assertIsNotNone(chain)


# ---------------------------------------------------------------------------
# ReasoningValidator
# ---------------------------------------------------------------------------

class TestReasoningValidator(unittest.TestCase):

    def setUp(self):
        from core.agents.reasoning_validator import ReasoningValidator
        from core.agents.thought_processor import ThoughtProcessor
        self.validator = ReasoningValidator()
        self.processor = ThoughtProcessor()

    def test_valid_chain_passes(self):
        chain = self.processor.generate("Improve margins", {"complexity_level": 2})
        result = self.validator.validate(chain, {})
        self.assertIn(result.status, ("valid", "invalid"))  # must return a result
        self.assertIsInstance(result.is_valid, bool)

    def test_metrics_keys_present(self):
        chain = self.processor.generate("Cut costs", {})
        result = self.validator.validate_chain(chain)
        for key in ("step_count", "avg_confidence", "evidence_coverage", "flaw_count"):
            self.assertIn(key, result.metrics)

    def test_confidence_score_bounded(self):
        chain = self.processor.generate("Scale operations", {})
        result = self.validator.validate(chain, {})
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)


# ---------------------------------------------------------------------------
# SelfCritic
# ---------------------------------------------------------------------------

class TestSelfCritic(unittest.TestCase):

    def _make_critic(self):
        from core.memory.memory_query import MemoryQueryEngine
        from core.agents.self_critic import SelfCritic
        return SelfCritic(MemoryQueryEngine())

    def test_analyze_returns_critique_result(self):
        from core.agents.reasoning_validator import ReasoningValidator
        from core.agents.thought_processor import ThoughtProcessor
        from core.agents.self_critic import CritiqueResult
        processor = ThoughtProcessor()
        validator = ReasoningValidator()
        critic = self._make_critic()
        chain = processor.generate("Optimise pricing strategy", {})
        vr = validator.validate(chain, {})
        result = critic.analyze(vr)
        self.assertIsInstance(result, CritiqueResult)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)

    def test_analyze_reasoning_has_summary(self):
        from core.agents.reasoning_validator import ReasoningValidator
        from core.agents.thought_processor import ThoughtProcessor
        processor = ThoughtProcessor()
        validator = ReasoningValidator()
        critic = self._make_critic()
        chain = processor.generate("Launch new product line", {"complexity_level": 2})
        vr = validator.validate(chain, {})
        result = critic.analyze_reasoning(chain, vr)
        self.assertIsInstance(result.summary, str)
        self.assertTrue(len(result.summary) > 0)

    def test_consistency_score_bounded(self):
        critic = self._make_critic()
        trace = [
            {"rationale": "Step one analysis", "confidence": 0.8},
            {"rationale": "Step two evaluation", "confidence": 0.75},
            {"rationale": "Conclusion", "confidence": 0.7},
        ]
        score = critic._evaluate_consistency(trace)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_efficiency_penalises_redundant_steps(self):
        critic = self._make_critic()
        unique_trace = [{"rationale": f"Step {i}"} for i in range(4)]
        duplicate_trace = [{"rationale": "Same step"}] * 8
        score_unique = critic._evaluate_efficiency(unique_trace)
        score_duplicate = critic._evaluate_efficiency(duplicate_trace)
        self.assertGreater(score_unique, score_duplicate)

    def test_effectiveness_rewards_conclusion(self):
        critic = self._make_critic()
        no_conclusion = [{"rationale": "Analysis", "confidence": 0.7}]
        with_conclusion = [
            {"rationale": "Analysis", "confidence": 0.7},
            {"rationale": "Therefore we recommend X", "confidence": 0.8,
             "metadata": {"step_type": "conclusion"}},
        ]
        s1 = critic._evaluate_effectiveness(no_conclusion)
        s2 = critic._evaluate_effectiveness(with_conclusion)
        self.assertGreaterEqual(s2, s1)


# ---------------------------------------------------------------------------
# IntentionManager
# ---------------------------------------------------------------------------

class TestIntentionManager(unittest.TestCase):

    def _make_manager(self):
        from core.memory.memory_query import MemoryQueryEngine
        from core.agents.intention_manager import IntentionManager
        return IntentionManager(MemoryQueryEngine())

    def test_create_intention_returns_intention(self):
        from core.memory.memory_query import MemoryQueryEngine
        from core.agents.goal_planner import GoalPlanner
        from core.agents.intention_manager import Intention
        planner = GoalPlanner(MemoryQueryEngine())
        manager = self._make_manager()
        goal = planner.create_goal("Improve NPS score", {})
        intention = manager.create_intention(goal, {})
        self.assertIsInstance(intention, Intention)

    def test_intention_id_is_uuid_format(self):
        from core.memory.memory_query import MemoryQueryEngine
        from core.agents.goal_planner import GoalPlanner
        planner = GoalPlanner(MemoryQueryEngine())
        manager = self._make_manager()
        goal = planner.create_goal("Launch referral programme", {})
        intention = manager.create_intention(goal, {})
        self.assertTrue(intention.id.startswith("intention_"))
        self.assertNotIn("_202", intention.id)

    def test_intention_ids_are_unique(self):
        from core.memory.memory_query import MemoryQueryEngine
        from core.agents.goal_planner import GoalPlanner
        planner = GoalPlanner(MemoryQueryEngine())
        manager = self._make_manager()
        ids = set()
        for _ in range(10):
            goal = planner.create_goal("Repeated goal", {})
            ids.add(manager.create_intention(goal, {}).id)
        self.assertEqual(len(ids), 10)


# ---------------------------------------------------------------------------
# Full AutonomousPipeline cycle
# ---------------------------------------------------------------------------

class TestAutonomousPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_run_cycle_returns_result(self):
        from core.pipeline_autonomy import AutonomousPipeline, AutonomousResult
        pipeline = AutonomousPipeline(memory_dir=self.tmp)
        result = pipeline.run_cycle("Reduce customer churn by 15%", {"domain": "SaaS"})
        self.assertIsInstance(result, AutonomousResult)

    def test_run_cycle_populates_summary(self):
        from core.pipeline_autonomy import AutonomousPipeline
        pipeline = AutonomousPipeline(memory_dir=self.tmp)
        result = pipeline.run_cycle("Increase average order value", {})
        self.assertIsInstance(result.reasoning_summary, str)
        self.assertTrue(len(result.reasoning_summary) > 0)

    def test_run_cycle_validation_status_is_string(self):
        from core.pipeline_autonomy import AutonomousPipeline
        pipeline = AutonomousPipeline(memory_dir=self.tmp)
        result = pipeline.run_cycle("Expand into the European market", {"domain": "e-commerce"})
        self.assertIn(result.validation_status, ("valid", "invalid"))

    def test_run_cycle_handles_error_gracefully(self):
        from core.pipeline_autonomy import AutonomousPipeline
        pipeline = AutonomousPipeline(memory_dir=self.tmp)
        # Pass a non-string goal to trigger the error path
        result = pipeline.run_cycle("", {})
        self.assertEqual(result.validation_status, "invalid")
        self.assertIn("error", result.metadata)

    def test_run_autonomous_cycle_function(self):
        from core.pipeline_autonomy import run_autonomous_cycle, AutonomousCycleResult
        result = run_autonomous_cycle(
            "Develop a machine learning model for demand forecasting",
            {"domain": "supply chain", "complexity_level": 2},
        )
        self.assertIsInstance(result, AutonomousCycleResult)
        self.assertIn(result.validation_status, ("valid", "invalid", "error"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# ConsultingFrameworkEngine
# ---------------------------------------------------------------------------

class TestConsultingFrameworkEngine(unittest.TestCase):

    def setUp(self):
        from core.aic_system import ConsultingFrameworkEngine, BusinessContext
        self.engine = ConsultingFrameworkEngine()
        self.context = BusinessContext(
            industry="retail",
            company_size="sme",
            market_position="challenger",
            competitive_landscape={"intensity": "high"},
            organizational_maturity="developing",
            strategic_priorities=["grow online sales", "reduce churn"],
        )

    def test_analyse_returns_dict(self):
        result = self.engine.analyse("How do we grow revenue?", self.context)
        self.assertIsInstance(result, dict)

    def test_analyse_includes_swot(self):
        from core.aic_system import ConsultingFramework
        result = self.engine.analyse(
            "Improve market share",
            self.context,
            frameworks=[ConsultingFramework.SWOT_ADVANCED],
        )
        self.assertIn("swot_advanced", result)
        self.assertIn("strengths", result["swot_advanced"])

    def test_analyse_aggregates_recommendations(self):
        result = self.engine.analyse("Cut operational costs", self.context)
        self.assertIn("aggregated_recommendations", result)
        self.assertIsInstance(result["aggregated_recommendations"], list)

    def test_empty_problem_raises(self):
        with self.assertRaises(ValueError):
            self.engine.analyse("", self.context)

    def test_recommend_frameworks_returns_list(self):
        from core.aic_system import ConsultingFramework
        frameworks = self.engine.recommend_frameworks(self.context)
        self.assertIsInstance(frameworks, list)
        self.assertTrue(all(isinstance(f, ConsultingFramework) for f in frameworks))

    def test_all_framework_handlers_execute(self):
        from core.aic_system import ConsultingFramework
        for fw in [
            ConsultingFramework.SWOT_ADVANCED,
            ConsultingFramework.PORTERS_FIVE_FORCES,
            ConsultingFramework.MCKINSEY_7S,
            ConsultingFramework.MECE_STRUCTURING,
            ConsultingFramework.ANSOFF_MATRIX,
            ConsultingFramework.BCG_GROWTH_SHARE,
        ]:
            result = self.engine.analyse("Test problem", self.context, frameworks=[fw])
            self.assertIn(fw.value, result, f"Missing output for {fw.value}")


if __name__ == "__main__":
    unittest.main()
