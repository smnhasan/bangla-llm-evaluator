from typing import Dict
import logging
from .config import Config
from .models.bangla_nlp import BanglaNLP

logging.basicConfig(level=Config.LOG_LEVEL)

class Metrics:
    def __init__(self):
        self.bangla_nlp = BanglaNLP()
        self.logger = logging.getLogger(__name__)

    def evaluate_fluency(self, conversation: Dict) -> float:
        """Evaluate conversation fluency in Bangla"""
        text = conversation.get('text', '')
        # Basic fluency check based on sentence structure and coherence
        sentences = self.bangla_nlp.tokenize_sentences(text)
        if not sentences:
            return 0.0
        return min(1.0, len(sentences) / 10.0)  # Simple heuristic

    def evaluate_tool_calling(self, conversation: Dict) -> float:
        """Evaluate tool calling performance"""
        tools_used = conversation.get('tools', [])
        expected_tools = conversation.get('expected_tools', [])
        if not expected_tools:
            return 1.0 if not tools_used else 0.5
        return len(set(tools_used) & set(expected_tools)) / len(expected_tools)

    def evaluate_edge_cases(self, conversation: Dict) -> float:
        """Evaluate handling of edge cases"""
        is_edge_case = conversation.get('is_edge_case', False)
        response = conversation.get('response', '')
        if is_edge_case and response:
            return 0.8 if self.bangla_nlp.is_coherent(response) else 0.2
        return 1.0

    def evaluate_instruction_adherence(self, conversation: Dict) -> float:
        """Evaluate adherence to special instructions"""
        instructions = conversation.get('instructions', '')
        response = conversation.get('response', '')
        if not instructions:
            return 1.0
        keywords = self.bangla_nlp.extract_keywords(instructions)
        return sum(1 for kw in keywords if kw in response) / len(keywords)

    def evaluate_task_accuracy(self, conversation: Dict) -> float:
        """Evaluate task execution accuracy"""
        expected_output = conversation.get('expected_output', '')
        actual_output = conversation.get('response', '')
        if not expected_output:
            return 1.0
        return self.bangla_nlp.calculate_similarity(expected_output, actual_output)
    