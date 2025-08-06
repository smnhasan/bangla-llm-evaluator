import json
import os
from typing import Dict
from .config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)

class Utils:
    @staticmethod
    def save_results(results: Dict, filename: str):
        """Save evaluation results to file"""
        output_path = os.path.join(Config.OUTPUT_DIR, filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"Results saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")

    @staticmethod
    def load_results(filename: str) -> Dict:
        """Load evaluation results from file"""
        input_path = os.path.join(Config.OUTPUT_DIR, filename)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Results file not found: {input_path}")
            return {}
        