from enum import Enum

class ConsultingRole(Enum):
    PROBLEM_STRUCTURER = "problem_structurer"
    HYPOTHESIS_GENERATOR = "hypothesis_generator" 
    DATA_INTERPRETER = "data_interpreter"
    STRATEGY_SYNTHESIZER = "strategy_synthesizer"
    EXECUTIVE_ADVISOR = "executive_advisor"

class ReasoningType(Enum):
    DEDUCTIVE = "deductive"      # From general to specific
    INDUCTIVE = "inductive"      # From specific to general  
    ABDUCTIVE = "abductive"      # Best explanation reasoning
    ANALOGICAL = "analogical"    # Pattern matching from past cases 