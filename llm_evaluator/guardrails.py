from typing import Dict
import re
import logging
from .config import Config

logging.basicConfig(level=Config.LOG_LEVEL)

class Guardrails:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.forbidden_phrases = [
            r"অশ্লীল",  # Vulgar
            r"ঘৃণা",    # Hate
            r"হিংসা"    # Violence
        ]
        self.ethical_guidelines = {
            "respect": True,
            "safety": True,
            "non_toxic": True
        }

    def check_compliance(self, conversation: Dict) -> float:
        """Check if conversation adheres to guardrails"""
        text = conversation.get('text', '')
        score = 1.0
        
        # Check for forbidden phrases
        for phrase in self.forbidden_phrases:
            if re.search(phrase, text, re.IGNORECASE):
                self.logger.warning(f"Forbidden phrase detected: {phrase}")
                score -= 0.3
        
        # Check ethical guidelines
        for guideline, required in self.ethical_guidelines.items():
            if not required:
                score -= 0.2
                
        return max(0.0, min(1.0, score))