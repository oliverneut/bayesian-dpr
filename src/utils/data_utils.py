from typing import Optional
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).parent.parent.parent

class DatasetConfig:
    def __init__(self, dataset_id: str):
        self.root_dir = PROJECT_ROOT / "data" / dataset_id
        self.dataset_id = dataset_id
        self.prepared_dir = self.create_prepared_dir()
        self.test_name = "test" if dataset_id in {"nq"} else "dev"
    
    def get_corpus_file(self) -> str:
        return self.root_dir / "corpus.jsonl"
    
    def get_queries_file(self) -> str:
        return self.root_dir / "queries.jsonl"
    
    def get_hard_negatives_file(self) -> Optional[str]:
        if self.dataset_id == "msmarco":
            return self.root_dir / "msmarco-hard-negatives.jsonl.gz"
        return None
    
    def get_query_file(self, split: str) -> str:
        return self.root_dir / "prepared" / f"queries-{split}.jsonl"
    
    def get_qrels_file(self, split: str) -> str:
        return self.root_dir / "qrels" / f"{split}.tsv"
    
    def create_prepared_dir(self) -> None:
        os.makedirs(self.root_dir / "prepared", exist_ok=True)
        return self.root_dir / "prepared"