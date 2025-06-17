import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import logging
from utils.indexing import FaissIndex
import wandb
from utils.evaluation import Evaluator
from vbll.layers.regression import VBLLReturn
from pathlib import Path
import csv
import os

from utils.model_utils import vbll_model_factory
from utils.data_utils import DatasetConfig
from prepare_data import DataWriter

PROJECT_ROOT = Path().resolve()


logger = logging.getLogger(__name__)


def generate_qrels(qrels_path: Path, qrels_dir: Path):
    with open(qrels_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        qrels = [['query-id', 'corpus-id', 'score']]
        for line in lines:
            tokens = line.strip().split()
            qry_id = tokens[0]
            doc_id = tokens[2]
            rel = int(tokens[3])
            qrels.append([qry_id, doc_id, rel])

    with open(qrels_dir/'dev.tsv', mode='w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(qrels)


def generate_queries(queries_path: Path, queries_dir: Path, split='clean'):
    with open(queries_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        
        queries = {}
        for line in lines:
            tokens = line.strip().split('\t')
            qry_id = tokens[0]
            qry_text = tokens[1]
            queries[qry_id] = {'query': qry_text}
    
    DataWriter.save_queries(queries, queries_dir, split)


def setup_data(data_cfg: DatasetConfig):
    qrels_path = data_cfg.root_dir / 'qrels.txt'
    cleans_path = data_cfg.root_dir / 'query.tsv'
    typos_path = data_cfg.root_dir / 'query.typo.tsv'
    qrels_dir = data_cfg.root_dir / 'qrels'
    
    os.makedirs(data_cfg.root_dir / 'qrels', exist_ok=True)
    os.makedirs(data_cfg.prepared_dir, exist_ok=True)

    if not Path(qrels_dir / 'dev.tsv').exists():
        generate_qrels(qrels_path, qrels_dir)

    if not Path(data_cfg.prepared_dir / 'queries-clean-dev.jsonl').exists():
        generate_queries(cleans_path, data_cfg.prepared_dir, split='clean')
        generate_queries(typos_path, data_cfg.prepared_dir, split='typo')


def load_embeddings(run_id: str, embs_dir: str):
    psg_embs = torch.load(embs_dir / f"{run_id}/embeddings.pt")
    psg_ids = torch.load(embs_dir / f"{run_id}/ids.pt")

    return psg_embs, psg_ids


def evaluate_model(model, tokenizer, index, psg_ids, queries, qrels, device, eval_mode="dpr", k=10):
    evaluator = Evaluator(tokenizer, model, eval_mode, device, index=index,
        metrics={"ndcg", "recip_rank"}, psg_ids=psg_ids)
    
    metrics = evaluator.evaluate_retriever(queries, qrels, k=10)
    
    logger.info(f"nDCG@{10}: {metrics[f"nDCG@{10}"]}")
    logger.info(f"MRR@{10}: {metrics[f"MRR@{10}"]}")


def main(model_id: str, run_id: str, embs_dir: str, eval_mode: str = "dpr"):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Run ID: {run_id}")
    logger.info("Setting up data...")
    data_cfg = DatasetConfig('dl-typo')
    setup_data(data_cfg)

    data_cfg = DatasetConfig('dl-typo')
    logger.info(f"Loading data")
    clean_queries = data_cfg.get_query_file(split='clean')
    logger.info(f"Clean queries loaded: {len(clean_queries)}")

    typo_queries = data_cfg.get_query_file(split='typo')
    logger.info(f"Typo queries loaded: {len(typo_queries)}")

    qrels = data_cfg.get_qrels_file(split='dev')

    psg_embs, psg_ids = load_embeddings(run_id, embs_dir)
    model, tokenizer = vbll_model_factory(model_id, device)
    index = FaissIndex.build(psg_embs)

    logger.info("Evaluating clean queries...")
    evaluate_model(model, tokenizer, index, psg_ids, clean_queries, qrels, device)

    logger.info("Evaluating typo queries...")
    evaluate_model(model, tokenizer, index, psg_ids, typo_queries, qrels, device)


if __name__ == '__main__':
    args = OmegaConf.load('config.yml')
    run_id = args.wandb.run_id
    embs_dir = args.eval.embs_dir

    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    model_id = config['model_name']
    main(model_id, run_id, embs_dir)
