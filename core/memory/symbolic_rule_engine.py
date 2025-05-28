from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from .decay_manager import DecayManager, DecayConfig
import logging
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class RuleConditionType(Enum):
    EQUALS = "equals"
    CONTAINS = "contains"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    REGEX_MATCH = "regex_match"
    SEMANTIC_MATCH = "semantic_match"
    CUSTOM = "custom"

@dataclass
class RuleCondition:
    """Defines a condition for rule matching"""
    condition_type: RuleConditionType
    field: str
    value: Any
    threshold: float = 0.0  # For semantic matching
    custom_evaluator: Optional[Callable[[Any], bool]] = None

@dataclass
class RuleAction:
    """Defines an action to take when rule matches"""
    action_type: str
    parameters: Dict[str, Any]
    priority: int = 0
    agent_id: Optional[str] = None

class SymbolicRule:
    """Represents a symbolic rule with conditions and actions"""
    def __init__(self,
                 name: str,
                 conditions: List[RuleCondition],
                 actions: List[RuleAction],
                 priority: int = 0):
        self.name = name
        self.conditions = conditions
        self.actions = actions
        self.priority = priority
        self.created_at = datetime.now()
        self.last_triggered = None
        self.trigger_count = 0
        self.metadata: Dict[str, Any] = {}

    def evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate all conditions against context"""
        for condition in self.conditions:
            if not self._evaluate_condition(condition, context):
                return False
        return True

    def _evaluate_condition(self, condition: RuleCondition, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        if condition.field not in context:
            return False

        value = context[condition.field]

        if condition.condition_type == RuleConditionType.EQUALS:
            return value == condition.value

        elif condition.condition_type == RuleConditionType.CONTAINS:
            return condition.value in value

        elif condition.condition_type == RuleConditionType.GREATER_THAN:
            return float(value) > float(condition.value)

        elif condition.condition_type == RuleConditionType.LESS_THAN:
            return float(value) < float(condition.value)

        elif condition.condition_type == RuleConditionType.SEMANTIC_MATCH:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb1 = model.encode([str(value)])[0]
            emb2 = model.encode([str(condition.value)])[0]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return similarity >= condition.threshold

        elif condition.condition_type == RuleConditionType.CUSTOM:
            if condition.custom_evaluator:
                return condition.custom_evaluator(value)
            return False

        return False

    def execute_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute all actions with context"""
        results = []
        for action in self.actions:
            try:
                result = self._execute_action(action, context)
                results.append({
                    'action': action.action_type,
                    'result': result,
                    'agent_id': action.agent_id,
                    'timestamp': datetime.now()
                })
            except Exception as e:
                logger.error(f"Error executing action {action.action_type}: {str(e)}")
                results.append({
                    'action': action.action_type,
                    'error': str(e),
                    'agent_id': action.agent_id,
                    'timestamp': datetime.now()
                })
        return results

    def _execute_action(self, action: RuleAction, context: Dict[str, Any]) -> Any:
        """Execute a single action"""
        # Implementation would vary based on action type
        return {'status': 'executed', 'parameters': action.parameters}

class SymbolicRuleEngine:
    """Engine for managing and executing symbolic rules"""
    def __init__(self, storage_path: Optional[Path] = None, decay_config: Optional[DecayConfig] = None):
        self.rules: Dict[str, SymbolicRule] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []
        self.storage_path = storage_path or Path("symbolic_rules.json")
        self.decay_manager = DecayManager(decay_config)
        
    def add_rule(self, rule: SymbolicRule):
        """Add a new rule to the engine"""
        self.rules[rule.name] = rule
        self.log_event('rule_added', {'rule_name': rule.name})

    def remove_rule(self, rule_name: str):
        """Remove a rule from the engine"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.log_event('rule_removed', {'rule_name': rule_name})

    def evaluate_context(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all rules against context"""
        results = []
        matched_rules = []

        # Find matching rules
        for rule in self.rules.values():
            if rule.evaluate_conditions(context):
                matched_rules.append(rule)

        # Sort by priority
        matched_rules.sort(key=lambda r: r.priority, reverse=True)

        # Execute actions for matched rules
        for rule in matched_rules:
            rule.last_triggered = datetime.now()
            rule.trigger_count += 1
            
            action_results = rule.execute_actions(context)
            results.extend(action_results)

            self.log_event('rule_triggered', {
                'rule_name': rule.name,
                'action_results': action_results
            })

        self.execution_history.append({
            'timestamp': datetime.now(),
            'context': context,
            'matched_rules': [r.name for r in matched_rules],
            'results': results
        })

        return results

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event with metadata"""
        self.logs.append({
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': data
        })

    def get_rule_stats(self) -> Dict[str, Any]:
        """Get statistics about rule execution"""
        stats = {
            'total_rules': len(self.rules),
            'total_executions': len(self.execution_history),
            'rule_triggers': {
                name: rule.trigger_count
                for name, rule in self.rules.items()
            },
            'last_execution': self.execution_history[-1]['timestamp']
            if self.execution_history else None
        }
        return stats

    def _apply_decay_cycle(self):
        """Apply decay to all rules"""
        if not self.decay_manager.should_check_decay():
            return
            
        # Calculate rule statistics
        stats = self._calculate_rule_stats()
        
        # Update decay parameters
        adaptive_params = self.decay_manager.get_adaptive_parameters(stats)
        self.decay_manager.update_decay_config(adaptive_params)
        
        # Apply decay to rules
        rules_to_remove = []
        for rule in self.rules.values():
            # Calculate new confidence
            new_confidence = self.decay_manager.apply_decay(
                rule.confidence,
                rule.created_at,
                rule.last_used,
                rule.use_count
            )
            
            # Check if rule should be pruned
            if self.decay_manager.should_prune(
                new_confidence,
                rule.last_used,
                rule.use_count
            ):
                rules_to_remove.append(rule)
            else:
                rule.confidence = new_confidence
                
        # Remove pruned rules
        for rule in rules_to_remove:
            del self.rules[rule.name]
            
        # Mark decay check complete
        self.decay_manager.mark_decay_check()
        
    def _calculate_rule_stats(self) -> Dict[str, Any]:
        """Calculate rule statistics"""
        if not self.rules:
            return {
                'avg_confidence': 0.5,
                'total_memories': 0,
                'avg_age': 0
            }
            
        total_rules = len(self.rules)
        avg_confidence = sum(rule.confidence for rule in self.rules.values()) / total_rules
        
        # Calculate average age in days
        total_age = sum(
            (datetime.now() - rule.created_at).total_seconds() / (24 * 3600)
            for rule in self.rules.values()
        )
        avg_age = total_age / total_rules
        
        return {
            'avg_confidence': avg_confidence,
            'total_memories': total_rules,
            'avg_age': avg_age
        }
        
    def reinforce_rule(self, rule: SymbolicRule, success_score: float):
        """Reinforce a rule based on successful use"""
        rule.confidence = self.decay_manager.apply_reinforcement(
            rule.confidence,
            success_score,
            rule.use_count
        )
        rule.last_used = datetime.now()
        rule.use_count += 1

    def save_rules(self):
        """Save rules to storage"""
        # Apply decay cycle before saving
        self._apply_decay_cycle()
        
        rules_data = []
        for rule in self.rules.values():
            rule_dict = {
                'name': rule.name,
                'conditions': [condition.__dict__ for condition in rule.conditions],
                'actions': [action.__dict__ for action in rule.actions],
                'priority': rule.priority,
                'created_at': rule.created_at.isoformat(),
                'last_used': rule.last_used.isoformat(),
                'use_count': rule.use_count,
                'confidence': rule.confidence
            }
            rules_data.append(rule_dict)
            
        with open(self.storage_path, 'w') as f:
            json.dump(rules_data, f, indent=2)

    def load_rules(self):
        """Load rules from storage"""
        if not self.storage_path.exists():
            return
            
        with open(self.storage_path) as f:
            rules_data = json.load(f)
            
        self.rules = {}
        for rule_dict in rules_data:
            rule = SymbolicRule(
                name=rule_dict['name'],
                conditions=[RuleCondition(**condition) for condition in rule_dict['conditions']],
                actions=[RuleAction(**action) for action in rule_dict['actions']],
                priority=rule_dict['priority']
            )
            rule.confidence = rule_dict['confidence']
            rule.created_at = datetime.fromisoformat(rule_dict['created_at'])
            rule.last_used = datetime.fromisoformat(rule_dict['last_used'])
            rule.use_count = rule_dict['use_count']
            self.rules[rule.name] = rule 