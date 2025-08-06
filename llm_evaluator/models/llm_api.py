import requests
from typing import Dict
import logging
from ..config import Config

logging.basicConfig(level=Config.LOG_LEVEL)

class LLMApi:
    def __init__(self, endpoint: str = Config.MODEL_API_ENDPOINT):
        self.endpoint = endpoint
        self.logger = logging.getLogger(__name__)

    def get_response(self, prompt: str) -> Dict:
        """Get response from LLM API"""
        try:
            response = requests.post(
                self.endpoint,
                json={"prompt": prompt},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return {"error": str(e)}
        