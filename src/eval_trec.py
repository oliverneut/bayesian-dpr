from omegaconf import OmegaConf
import logging
import torch
from utils.model_utils import vbll_model_factory, model_factory
from utils.indexing import FaissIndex
from pytrec_eval import RelevanceEvaluator
from collections import defaultdict
import wandb
import numpy as np
from vbll.layers.regression import VBLLReturn
import os
import ir_datasets
from tqdm import tqdm


logger = logging.getLogger(__name__)


def evaluate_trec(model, tokenizer, index, psg_ids, data, device):
    run = defaultdict(dict)
    trec_scores = defaultdict(dict)
    with torch.no_grad():
        for qry_id, qry in tqdm(data.queries, desc="Evaluating TREC-DL queries"):
            qry_enc = tokenizer(qry, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)

            qry_emb = model(qry_enc)
            if isinstance(qry_emb, VBLLReturn):
                qry_emb = qry_emb.predictive.loc

            scores, indices = index.search(qry_emb, k=10)
            psg_indices = [psg_ids[idx] for idx in indices[0]]
            for score, psg_id in zip(scores[0], psg_indices):
                run[qry_id][psg_id] = float(score)
        
        evaluator = RelevanceEvaluator(data.qrels_dict(), {"ndcg", "recip_rank"})
        results = evaluator.evaluate(run)

        for qry_id, metrics in results.items():
            trec_scores[qry_id]['ndcg'] = metrics['ndcg']
            trec_scores[qry_id]['mrr'] = metrics['recip_rank']

        ndcg_scores = [trec_scores[qry_id]['ndcg'] for qry_id in trec_scores]
        mrr_scores = [trec_scores[qry_id]['mrr'] for qry_id in trec_scores]
        logger.info(f"nDCG: {np.mean(ndcg_scores)}")
        logger.info(f"MRR: {np.mean(mrr_scores)}")


def main(model_name: str, vbll: bool, run_id: str):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(f"Run ID: {run_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    save_dir = f"output/models/{run_id}"
    model_path = f"{save_dir}/model.pt"

    if vbll:
        tokenizer, model = vbll_model_factory(model_name, device)
    else:
        tokenizer, model = model_factory(model_name, device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if os.path.exists(f"{save_dir}/psg_embs.pt") and os.path.exists(f"{save_dir}/psg_ids.pt"):
        logger.info("Loading precomputed embeddings and IDs from disk.")
        psg_embs = torch.load(f"{save_dir}/psg_embs.pt", map_location=device)
        psg_ids = torch.load(f"{save_dir}/psg_ids.pt")

        if psg_embs.dim() == 3:
            logger.info("Reshaping embeddings from 3D to 2D.")
            psg_embs = psg_embs[:,0]
    else:
        logger.info("No precomputed embeddings found. Please run the encoding script first.")
        return
    
    index = FaissIndex.build(psg_embs)

    dataset_name = "trec-dl-2019"
    trec_dl_19 = ir_datasets.load(f"msmarco-passage/{dataset_name}/judged")
    evaluate_trec(model, tokenizer, index, psg_ids, trec_dl_19, device)

    dataset_name = "trec-dl-2020"
    trec_dl_20 = ir_datasets.load(f"msmarco-passage/{dataset_name}/judged")
    evaluate_trec(model, tokenizer, index, psg_ids, trec_dl_20, device)


if __name__ == '__main__':
    args = OmegaConf.load('config.yml')
    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    model_id = config['model_name']
    vbll = config['knowledge_distillation']
    main(model_id, vbll, args.wandb.run_id)