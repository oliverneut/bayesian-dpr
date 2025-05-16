import torch
from scipy.stats import pearsonr
import torch
import wandb
from omegaconf import OmegaConf
from types import SimpleNamespace
from indexing import FaissIndex
from pytrec_eval import RelevanceEvaluator
from data_loaders import get_qrels
from torch.utils.data import DataLoader
from data_loaders import EmbeddingDataset
from utils.data_utils import DatasetConfig
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def uncertainty_score(cov):
    return torch.linalg.vector_norm(cov, ord=2, dim=2)

def main(args, run_id: str, data_cfg: DatasetConfig):
    model_dir = f"output/models/{run_id}"
    psg_embs_path = f"{model_dir}/psg_embs.pt"
    psg_ids_path = f"{model_dir}/psg_ids.pt"
    qry_embs_path = f"{model_dir}/qry_embs.pt"
    qry_ids_path = f"{model_dir}/qry_ids.pt"

    psg_embs = torch.load(psg_embs_path)
    psg_ids = torch.load(psg_ids_path)

    qry_dataset = EmbeddingDataset(qry_embs_path, qry_ids_path)
    qry_dataloader = DataLoader(qry_dataset, batch_size=16, shuffle=False)

    index = FaissIndex.build(psg_embs[:,0,:])

    qrels = get_qrels(data_cfg.get_qrels_file(split=data_cfg.test_name))

    uncertainty_scores = []
    ndcg_scores = []
    mrr_scores = []
    
    with torch.no_grad():
        for qry_embs, qry_ids in tqdm(qry_dataloader, desc="Computing QPP scores"):
            mean, cov = qry_embs[:,0,:], qry_embs[:,1:,:]
            run = {}
            uncertainty = uncertainty_score(cov)
            uncertainty_scores += uncertainty.squeeze().tolist()

            scores, indices = index.search(mean, k=10)

            batch_psg_indices = [[psg_ids[idx] for idx in batch_indices] for batch_indices in indices]
            
            for qry_id, query_scores, query_indices in zip(qry_ids, scores, batch_psg_indices):
                    run[qry_id] = {}
                    for score, psg_id in zip(query_scores, query_indices):
                        run[qry_id][psg_id] = float(score)

            evaluator = RelevanceEvaluator(qrels, {"ndcg", "recip_rank"})
            results = evaluator.evaluate(run)

            for _, metrics in results.items():
                 ndcg_scores.append(metrics['ndcg'])
                 mrr_scores.append(metrics['recip_rank'])

    ndcg_corr =  pearsonr(uncertainty_scores, ndcg_scores)
    mrr_corr = pearsonr(uncertainty_scores, mrr_scores)

    logger.info(f"nDCG Correlation: {ndcg_corr}")
    logger.info(f"MRR Correlation: {mrr_corr}")

if __name__ == '__main__':
    args = OmegaConf.load('src/utils/config.yml')
    data_cfg = DatasetConfig(args.prepare_data.dataset_id)
    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    params = SimpleNamespace(**config)
    main(params, args.wandb.run_id, data_cfg)