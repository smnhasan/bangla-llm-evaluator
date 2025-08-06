import os
import pandas as pd


class DatasetLoader:
    def __init__(self, base_data_dir: str = None):
        if base_data_dir is None:
            base_data_dir = os.path.join(os.path.dirname(__file__), "csv")
        self.data_dir = base_data_dir
        self.datasets = {}
        self.required_columns = ["input_text", "reference"]
        self._load_all_datasets()

    def _load_all_datasets(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".csv"):
                path = os.path.join(self.data_dir, filename)
                dataset_name = os.path.splitext(filename)[0]
                try:
                    df = pd.read_csv(path)
                    if self._validate_columns(df, filename):
                        self.datasets[dataset_name] = df
                except Exception as e:
                    print(f"[ERROR] Failed to load '{filename}': {e}")

    def _validate_columns(self, df: pd.DataFrame, filename: str) -> bool:
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            print(f"[WARNING] '{filename}' is missing required columns: {missing}")
            return False
        return True

    def get_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
        return self.datasets[name]

    def list_datasets(self) -> list:
        return list(self.datasets.keys())

