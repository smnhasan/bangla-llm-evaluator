import json
import logging
from typing import List, Dict
from .config import Config

logging.basicConfig(level=Config.LOG_LEVEL)

class DatasetLoader:
    def __init__(self, dataset_path: str = Config.DATASET_PATH):
        self.dataset_path = dataset_path
        self.logger = logging.getLogger(__name__)

    def load_dataset(self) -> List[Dict]:
        """Load Bangla conversation dataset from JSONL file"""
        try:
            dataset = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
            self.logger.info(f"Loaded {len(dataset)} conversation samples")
            return dataset
        except FileNotFoundError:
            self.logger.error(f"Dataset file not found: {self.dataset_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON format in dataset")
            raise

    def get_evaluation_samples(self, dimension: str) -> List[Dict]:
        """Get samples specific to an evaluation dimension"""
        dataset = self.load_dataset()
        return [sample for sample in dataset if sample.get('dimension') == dimension]
