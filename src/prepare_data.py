from omegaconf import OmegaConf
import gzip
import json
import os
from tqdm import tqdm
import random
from typing import Dict, Set, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from utils.config import (
    TOTAL_DOCUMENTS,
    CE_SCORE_MARGIN,
    HARD_NEGATIVES,
    PREPARED_DIR
)
from data_loaders import get_queries, get_corpus, get_qrels

@dataclass
class QueryData:
    qid: int
    query: str
    pos_pids: list
    neg_pids: list

class DataWriter:
    @staticmethod
    def save_triplets(data: Dict, data_dir: Path, split: str) -> None:
        """Save query triplets (query, positive passages, negative passages) to file."""
        with open(data_dir / f'queries-{split}.jsonl', 'wt', encoding='utf8') as f_out:
            for qid, query_data in tqdm(data.items(), desc=f"Saving {split} triplets"):
                json.dump({
                    "qid": qid,
                    "query": query_data["query"],
                    "pos": query_data["pos"],
                    "neg": query_data["neg"]
                }, f_out)
                f_out.write("\n")

    @staticmethod
    def save_queries(data: Dict, data_dir: Path, split: str) -> None:
        """Save queries to file."""
        with open(data_dir / f'queries-{split}.jsonl', 'wt', encoding='utf8') as f_out:
            for qid, query_data in tqdm(data.items(), desc=f"Saving {split} queries"):
                json.dump({"qid": qid, "query": query_data["query"]}, f_out)
                f_out.write("\n")

    @staticmethod
    def save_corpus(data: Dict, pids: Set, data_dir: Path, split: str) -> None:
        """Save corpus passages to file."""
        with open(data_dir / f'corpus-{split}.jsonl', 'wt', encoding='utf8') as f_out:
            for pid, text in tqdm(data.items(), desc=f"Saving {split} corpus"):
                if pid in pids:
                    json.dump({"_id": pid, "text": text}, f_out)
                    f_out.write("\n")

class DataProcessor:
    def __init__(self, ce_score_margin: float):
        self.ce_score_margin = ce_score_margin

    def process_query(self, data: Dict, queries: Dict) -> Optional[QueryData]:
        """Process a single query's data to extract positive and negative passages."""
        qid = int(data["qid"])
        
        # Get positive passages and compute threshold
        pos_pids = [item["pid"] for item in data["pos"]]
        if not pos_pids:
            return None
            
        pos_min_ce_score = min(item["ce-score"] for item in data["pos"])
        ce_score_threshold = pos_min_ce_score - self.ce_score_margin
        
        # Get negative passages below threshold
        neg_pids = {
            item["pid"] 
            for system_negs in data["neg"].values()
            for item in system_negs 
            if item["ce-score"] <= ce_score_threshold
        }
        
        if not neg_pids:
            return None
            
        return QueryData(
            qid=qid,
            query=queries[str(qid)],
            pos_pids=pos_pids,
            neg_pids=list(neg_pids)
        )

    def build_dataset(self, lines: list, queries: Dict) -> Tuple[Dict, Set]:
        """Build dataset from lines of query data."""
        dataset_pids = set()
        dataset = {}
        
        for line in tqdm(lines, desc="Processing queries"):
            data = json.loads(line)
            processed = self.process_query(data, queries)
            
            if processed is None:
                continue
                
            dataset_pids.update(processed.pos_pids)
            dataset_pids.update(processed.neg_pids)
            
            dataset[processed.qid] = {
                "query": processed.query,
                "pos": processed.pos_pids,
                "neg": processed.neg_pids
            }
        
        return dataset, dataset_pids

def prepare_data(queries: Dict, val_size: int) -> None:
    """Prepare training and validation datasets."""
    # Create output directory
    os.makedirs(PREPARED_DIR, exist_ok=True)
    
    # Initialize processor and writer
    processor = DataProcessor(CE_SCORE_MARGIN)
    writer = DataWriter()
    
    # Load and shuffle data
    with gzip.open(HARD_NEGATIVES, 'rt', encoding='utf8') as f_in:
        lines = f_in.readlines()
    random.shuffle(lines)
    
    # Split into train and validation
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]
    
    # Process datasets
    train_data, _ = processor.build_dataset(train_lines, queries)
    val_data, val_pids = processor.build_dataset(val_lines, queries)
    
    # Save datasets
    writer.save_triplets(train_data, PREPARED_DIR, "train")
    writer.save_queries(val_data, PREPARED_DIR, "val")
    
    # Save corpus
    corpus = get_corpus()
    writer.save_corpus(corpus, val_pids, PREPARED_DIR, "val")

def prepare_test_queries(queries: Dict) -> None:
    """Prepare test queries dataset."""
    qrels = get_qrels(split="dev")
    test_queries = {
        qid: {"query": queries[qid]}
        for qid, rels in qrels.items()
        if len(rels) > 0
    }
    
    DataWriter.save_queries(test_queries, PREPARED_DIR, "dev")

if __name__ == '__main__':
    args = OmegaConf.load('src/utils/config.yml').prepare_data
    queries = get_queries()
    prepare_data(queries, val_size=args.val_size)
    prepare_test_queries(queries)