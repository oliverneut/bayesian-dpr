import argparse
import gzip
import json
import os
from tqdm import tqdm
import random
from typing import Dict
from pathlib import PosixPath
from utils.config import (
    TOTAL_DOCUMENTS,
    VALIDATION_RATIO,
    CE_SCORE_MARGIN,
    HARD_NEGATIVES,
    PREPARED_DIR
)

from data_loaders import get_queries, get_corpus


def save_queries(data: Dict, data_dir: PosixPath, split: str):
    with open(f'{data_dir}/queries-{split}.jsonl', 'wt', encoding='utf8') as f_out:
        for k, v in tqdm(data.items(), total=len(data.keys())):
            json.dump({"qid": k, "query": v["query"], "pos": v["pos"], "neg": v["neg"]}, f_out)
            f_out.write("\n")


def save_corpus(data: Dict, pids: set, data_dir: PosixPath, split: str):
    with open(f'{data_dir}/corpus-{split}.jsonl', 'wt', encoding='utf8') as f_out:
        for k, v in tqdm(data.items(), total=len(data.keys())):
            if k in pids:
                json.dump({"pid": k, "text": v}, f_out)
                f_out.write("\n")


def get_N_sample_ids(total: int=TOTAL_DOCUMENTS, N: int=200):
    sampled_ids = random.sample(range(total), N)
    num_validation = int(VALIDATION_RATIO * N)
    random.shuffle(sampled_ids)

    train_ids = sampled_ids[num_validation:]

    return train_ids, sampled_ids


def prepare_data(N: int):
    count = 0
    train_ids, sampled_ids = get_N_sample_ids(total=TOTAL_DOCUMENTS, N=N)
    train_pids, val_pids = set(), set()

    queries = get_queries()

    train_data, val_data = {}, {}

    with gzip.open(HARD_NEGATIVES, 'rt', encoding='utf8') as f_in:
        for idx, line in tqdm(enumerate(f_in), total=TOTAL_DOCUMENTS):
            data = json.loads(line)
            qid = int(data["qid"])

            if idx in sampled_ids:
                count += 1
                pos_pids = [item["pid"] for item in data["pos"]]
                pos_min_ce_score = min([item["ce-score"] for item in data["pos"]])
                ce_score_threshold = pos_min_ce_score - CE_SCORE_MARGIN

                neg_pids = set()

                for system_negs in data["neg"].values():
                    for item in system_negs:
                        if item["ce-score"] <= ce_score_threshold:
                            neg_pids.add(item["pid"])
                
                if len(pos_pids) > 0 and len(neg_pids) > 0:
                    if idx in train_ids:
                        train_pids.update(pos_pids)
                        train_pids.update(neg_pids)
                        train_data[qid] = {"query": queries[str(qid)], "pos": pos_pids, "neg": list(neg_pids)}
                    else:
                        val_pids.update(pos_pids)
                        val_pids.update(neg_pids)
                        val_data[qid] = {"query": queries[str(qid)], "pos": pos_pids, "neg": list(neg_pids)}

    os.makedirs(PREPARED_DIR, exist_ok=True)
    save_queries(train_data, PREPARED_DIR, "train")
    save_queries(val_data, PREPARED_DIR, "val")

    corpus = get_corpus()

    save_corpus(corpus, val_pids, PREPARED_DIR, "val")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=TOTAL_DOCUMENTS)
    args = parser.parse_args()
    prepare_data(N=200)