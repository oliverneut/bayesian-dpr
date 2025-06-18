from omegaconf import OmegaConf
import torch
import logging
from utils.indexing import FaissIndex
from utils.run_utils import RunConfig
from utils.model_utils import get_model_from_run
from utils.data_utils import DatasetConfig
from utils.embedding_utils import has_embeddings, load_embeddings, get_embeddings
import wandb
from utils.evaluation import Evaluator
from vbll.layers.regression import VBLLReturn
from pathlib import Path
from utils.data_loaders import get_query_dataloader, get_qrels
import csv
import os
from prepare_data import DataWriter


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
    
    os.makedirs(qrels_dir, exist_ok=True)
    os.makedirs(data_cfg.prepared_dir, exist_ok=True)

    if not Path(qrels_dir / 'dev.tsv').exists():
        generate_qrels(qrels_path, qrels_dir)

    if not Path(data_cfg.prepared_dir / 'queries-clean.jsonl').exists():
        generate_queries(cleans_path, data_cfg.prepared_dir, split='clean')
        generate_queries(typos_path, data_cfg.prepared_dir, split='typo')


def evaluate_model(model, tokenizer, index, psg_ids, queries, qrels, device, rel_mode="dpr", k=10):
    evaluator = Evaluator(tokenizer, model, device, rel_mode, index=index,
        metrics={"ndcg", "recip_rank"}, psg_ids=psg_ids)
    
    metrics = evaluator.evaluate_retriever(queries, qrels, k=10)
    
    logger.info(f"nDCG@{10}: {metrics[f"nDCG@{10}"]}")
    logger.info(f"MRR@{10}: {metrics[f"MRR@{10}"]}")


def main(run_cfg: RunConfig, embs_dir: str, rel_mode: str = "dpr"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Setting up data...")
    data_cfg = DatasetConfig('dl-typo')
    setup_data(data_cfg)

    tokenizer, model = get_model_from_run(run_cfg, device)

    logger.info(f"Loading data")
    clean_queries = get_query_dataloader(data_cfg.get_query_file(split='clean'))
    typo_queries = get_query_dataloader(data_cfg.get_query_file(split='typo'))
    qrels =  get_qrels(data_cfg.get_qrels_file(split='dev'))

    if has_embeddings(run_cfg, data_cfg, embs_dir):
        psg_embs, psg_ids = load_embeddings(run_cfg, data_cfg, embs_dir, rel_mode, device)
    else:
        logger.info("No precomputed embeddings found. Please run the eval_retriever script first.")
        return

    logger.info("Building the index")
    index = FaissIndex.build(psg_embs)

    logger.info("Evaluating clean queries...")
    evaluate_model(model, tokenizer, index, psg_ids, clean_queries, qrels, device, rel_mode)
    logger.info("-" * 50)

    logger.info("Evaluating typo queries...")
    evaluate_model(model, tokenizer, index, psg_ids, typo_queries, qrels, device, rel_mode)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info("OOD-2 experiment")
    args = OmegaConf.load('config.yml')

    run_cfg = RunConfig(args)

    logger.info(f"Run ID: {args.wandb.run_id}")
    logger.info(f"Dataset id: {args.eval.dataset_id}")
    logger.info(f"Relevance mode: {args.eval.rel_mode}")

    main(run_cfg, embs_dir=args.eval.embs_dir, rel_mode=args.eval.rel_mode)
