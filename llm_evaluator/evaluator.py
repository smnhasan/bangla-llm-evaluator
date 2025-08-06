from typing import Dict, List
from .metrics import Metrics
from .scoring import Scorer
from .models.bangla_nlp import BanglaNLP
from .guardrails import Guardrails
from .config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)

class LLMEvaluator:
    def __init__(self):
        self.metrics = Metrics()
        self.scorer = Scorer()
        self.bangla_nlp = BanglaNLP()
        self.guardrails = Guardrails()
        self.logger = logging.getLogger(__name__)

    def evaluate(self, conversation: Dict) -> Dict:
        """Evaluate a single conversation across all dimensions"""
        results = {}
        for dimension in Config.EVALUATION_DIMENSIONS:
            score = self.evaluate_dimension(conversation, dimension)
            results[dimension] = score
        final_score = self.scorer.compute_weighted_score(results)
        results['final_score'] = final_score
        return results

    def evaluate_dimension(self, conversation: Dict, dimension: str) -> float:
        """Evaluate a specific dimension of the conversation"""
        if dimension == "conversation_fluency":
            return self.metrics.evaluate_fluency(conversation)
        elif dimension == "tool_calling_performance":
            return self.metrics.evaluate_tool_calling(conversation)
        elif dimension == "guardrails_compliance":
            return self.guardrails.check_compliance(conversation)
        elif dimension == "edge_case_handling":
            return self.metrics.evaluate_edge_cases(conversation)
        elif dimension == "special_instruction_adherence":
            return self.metrics.evaluate_instruction_adherence(conversation)
        elif dimension == "language_proficiency":
            return self.bangla_nlp.evaluate_proficiency(conversation)
        elif dimension == "task_execution_accuracy":
            return self.metrics.evaluate_task_accuracy(conversation)
        return 0.0
    