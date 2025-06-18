from omegaconf import OmegaConf
from utils.run_utils import RunConfig
from utils.data_utils import DatasetConfig
from utils.embedding_utils import has_embeddings, load_embeddings
import logging
import torch
from utils.model_utils import get_model_from_run
from utils.indexing import FaissIndex
from pytrec_eval import RelevanceEvaluator
from collections import defaultdict
from utils.evaluation import Evaluator
import numpy as np
from vbll.layers.regression import VBLLReturn
import ir_datasets
from tqdm import tqdm


logger = logging.getLogger(__name__)


def evaluate_trec(model, tokenizer, index, psg_ids, queries, device):
    run = defaultdict(dict)

    with torch.no_grad():
        for qry_id, qry in tqdm(queries, desc="Evaluating TREC-DL queries"):
            qry_enc = tokenizer(qry, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)

            qry_emb = model(qry_enc)
            if isinstance(qry_emb, VBLLReturn):
                qry_emb = qry_emb.predictive.mean

            scores, indices = index.search(qry_emb, k=10)
            psg_indices = [psg_ids[idx] for idx in indices[0]]
            for score, psg_id in zip(scores[0], psg_indices):
                run[qry_id][psg_id] = float(score)
        
    return run


def calculate_metrics(run, qrels, k=10):
    evaluator = RelevanceEvaluator(qrels, {"ndcg", "recip_rank"})
    results = evaluator.evaluate(run)

    ndcg_at_k = []
    mrr_at_k = []
    for _, metrics in results.items():
        ndcg_at_k.append(metrics["ndcg"])
        mrr_at_k.append(metrics["recip_rank"])
    results_agg = {
        f"nDCG@{k}": float(np.mean(ndcg_at_k)),
        f"MRR@{k}": float(np.mean(mrr_at_k)),
    }
    logger.info(results_agg)
    return results_agg


def main(run_cfg: RunConfig, data_cfg: DatasetConfig, embs_dir: str, rel_mode: str = "dpr"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer, model = get_model_from_run(run_cfg, device)

    if has_embeddings(run_cfg, data_cfg, embs_dir):
        psg_embs, psg_ids = load_embeddings(run_cfg, data_cfg, embs_dir, rel_mode, device)
    else:
        logger.info("No precomputed embeddings found. Please run the eval_retriever script first.")
        return
    
    index = FaissIndex.build(psg_embs)

    for dataset_name in ["trec-dl-2019", "trec-dl-2020"]:
        trec_dl = ir_datasets.load(f"msmarco-passage/{dataset_name}")
        logger.info(f"Evaluating {dataset_name} dataset...")
        run = evaluate_trec(model, tokenizer, index, psg_ids, trec_dl.queries, device)

        qrels = defaultdict(dict)
        for qrel in trec_dl.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        metrics = calculate_metrics(run, trec_dl.qrels_dict(), k=10)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info("Eval trec experiment")
    args = OmegaConf.load('config.yml')

    run_cfg = RunConfig(args)
    data_cfg = DatasetConfig(args.eval.dataset_id)

    logger.info(f"Run ID: {args.wandb.run_id}")
    assert args.eval.dataset_id == 'msmarco'
    logger.info(f"Dataset id: {args.eval.dataset_id}")

    main(run_cfg, data_cfg, embs_dir=args.eval.embs_dir)