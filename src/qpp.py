import torch
from scipy.stats import pearsonr
import torch
import wandb
from omegaconf import OmegaConf
from types import SimpleNamespace
from indexing import FaissIndex
from utils.model_utils import vbll_model_factory
from pytrec_eval import RelevanceEvaluator
from data_loaders import get_qrels
from torch.utils.data import DataLoader
from data_loaders import EmbeddingDataset
from utils.data_utils import DatasetConfig
from tqdm import tqdm
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def uncertainty_score(cov):
    return torch.linalg.vector_norm(cov, ord=2, dim=1)

def main(args: SimpleNamespace, run_id: str, data_cfg: DatasetConfig):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset id: {data_cfg.dataset_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = f"output/models/{run_id}"
    psg_embs_path = f"{model_dir}/psg_embs.pt"
    psg_ids_path = f"{model_dir}/psg_ids.pt"
    qry_embs_path = f"{model_dir}/qry_embs.pt"
    qry_ids_path = f"{model_dir}/qry_ids.pt"

    psg_embs = torch.load(psg_embs_path)
    psg_ids = torch.load(psg_ids_path)

    model_path = f"{model_dir}/model.pt"

    _, model = vbll_model_factory(args.model_name, 1, args.parameterization, args.prior_scale, args.wishart_scale, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    noise = model.vbll_layer.noise()

    qry_dataset = EmbeddingDataset(qry_embs_path, qry_ids_path)
    qry_dataloader = DataLoader(qry_dataset, batch_size=16, shuffle=False)

    index = FaissIndex.build(psg_embs[:,0,:])

    qrels = get_qrels(data_cfg.get_qrels_file(split=data_cfg.test_name))

    uncertainty_scores = []
    ndcg_scores = []
    mrr_scores = []
    run = {}
    qpp_scores = defaultdict(dict)

    with torch.no_grad():
        for qry_embs, qry_ids in tqdm(qry_dataloader, desc="Computing QPP scores"):
            mean, cov = qry_embs[:,0,:], qry_embs[:,1,:]

            cov = cov - noise.scale
            u_scores = uncertainty_score(cov).tolist()
            uncertainty_scores += u_scores
            for qry_id, uncertainty in zip(qry_ids, u_scores):
                qpp_scores[qry_id]['uncertainty'] = uncertainty

            scores, indices = index.search(mean, k=10)

            batch_psg_indices = [[psg_ids[idx] for idx in batch_indices] for batch_indices in indices]
            
            for qry_id, query_scores, query_indices in zip(qry_ids, scores, batch_psg_indices):
                    run[qry_id] = {}
                    for score, psg_id in zip(query_scores, query_indices):
                        run[qry_id][psg_id] = float(score)

        evaluator = RelevanceEvaluator(qrels, {"ndcg", "recip_rank"})
        results = evaluator.evaluate(run)

        for qry_id, metrics in results.items():
            qpp_scores[qry_id]['ndcg'] = metrics['ndcg']
            qpp_scores[qry_id]['mrr'] = metrics['recip_rank']

    uncertainty_scores = [qpp_scores[qry_id]['uncertainty'] for qry_id in qpp_scores]
    ndcg_scores = [qpp_scores[qry_id]['ndcg'] for qry_id in qpp_scores]
    mrr_scores = [qpp_scores[qry_id]['mrr'] for qry_id in qpp_scores]

    ndcg_corr =  pearsonr(uncertainty_scores, ndcg_scores)
    mrr_corr = pearsonr(uncertainty_scores, mrr_scores)

    logger.info(f"nDCG Correlation: {ndcg_corr}")
    logger.info(f"MRR Correlation: {mrr_corr}")

if __name__ == '__main__':
    args = OmegaConf.load('config.yml')
    data_cfg = DatasetConfig(args.prepare_data.dataset_id)
    main(args.train, args.wandb.run_id, data_cfg)