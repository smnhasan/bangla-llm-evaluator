from typing import Dict
from .config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)

class Scorer:
    def __init__(self):
        self.weights = Config.SCORING_WEIGHTS
        self.logger = logging.getLogger(__name__)

    def compute_weighted_score(self, results: Dict) -> float:
        """Compute weighted average score across dimensions"""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, score in results.items():
            weight = self.weights.get(dimension, 0.0)
            total_score += score * weight
            total_weight += weight
            
        if total_weight == 0:
            self.logger.warning("No valid weights found for scoring")
            return 0.0
            
        return total_score / total_weight
    