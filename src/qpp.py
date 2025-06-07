import torch
from scipy.stats import pearsonr, kendalltau
import torch
import wandb
import numpy as np
from omegaconf import OmegaConf
from types import SimpleNamespace
from indexing import FaissIndex
from utils.model_utils import vbll_model_factory
from pytrec_eval import RelevanceEvaluator
from utils.data_utils import DatasetConfig
from tqdm import tqdm
from collections import defaultdict
import logging
import ir_datasets
import json

logger = logging.getLogger(__name__)


def uncertainty_score(scale, unc_method="norm"):
    cov = torch.diag(scale.squeeze())

    if unc_method == "norm":
        return torch.linalg.norm(cov)
    elif unc_method == "trace":
        return torch.trace(cov)
    elif unc_method == "det":
        _, logdet = torch.linalg.slogdet(cov)
        return logdet
    elif unc_method == "entropy":
        d = cov.size(0)
        _, logdet = torch.linalg.slogdet(cov)
        return 0.5 * d * torch.log(torch.tensor(2 * torch.pi * torch.e)) + 0.5 * logdet
    else:
        raise ValueError(f"Unknown uncertainty method: {unc_method}")


def infer_embedding(model, tokenizer, query, device):
    qry_enc = tokenizer(query, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)
    qry_emb = model(qry_enc).predictive

    return qry_emb


def qpp(data, model, tokenizer, index, psg_ids, device, unc_method="norm"):
    run = defaultdict(dict)
    qpp_scores = defaultdict(dict)

    with torch.no_grad():
        for qry_id, qry in tqdm(data.queries, desc="Computing QPP scores"):
            emb = infer_embedding(model, tokenizer, qry, device)
            
            uncertainty = uncertainty_score(emb.scale, unc_method).item()

            qpp_scores[qry_id]['uncertainty'] = uncertainty

            scores, indices = index.search(emb.loc, k=10)

            psg_indices = [psg_ids[idx] for idx in indices[0]]
        
            for score, psg_id in zip(scores[0], psg_indices):
                run[qry_id][psg_id] = float(score)
        
        evaluator = RelevanceEvaluator(data.qrels_dict(), {"ndcg", "recip_rank"})
        results = evaluator.evaluate(run)
        with open('results.json', 'w', encoding='utf8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        for qry_id, metrics in results.items():
            qpp_scores[qry_id]['ndcg'] = metrics['ndcg']
            qpp_scores[qry_id]['mrr'] = metrics['recip_rank']

    uncertainty_scores = [qpp_scores[qry_id]['uncertainty'] for qry_id in qpp_scores]
    ndcg_scores = [qpp_scores[qry_id]['ndcg'] for qry_id in qpp_scores]
    mrr_scores = [qpp_scores[qry_id]['mrr'] for qry_id in qpp_scores]

    logger.info(f"nDCG Pearson Correlation: {pearsonr(uncertainty_scores, ndcg_scores)}")
    logger.info(f"MRR Pearson Correlation: {pearsonr(uncertainty_scores, mrr_scores)}")
    logger.info(f"nDCG Kendall Tau: {kendalltau(uncertainty_scores, ndcg_scores)}")
    logger.info(f"MRR Kendall Tau: {kendalltau(uncertainty_scores, mrr_scores)}")
    logger.info(f"nDCG: {np.mean(ndcg_scores)}")
    logger.info(f"MRR: {np.mean(mrr_scores)}")


def main(args: SimpleNamespace, run_id: str, unc_method="norm"):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(f"Run ID: {run_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = f"output/models/{run_id}"
    psg_embs = torch.load(f"{model_dir}/psg_embs.pt")
    psg_ids = torch.load(f"{model_dir}/psg_ids.pt")

    model_path = f"{model_dir}/model.pt"

    tokenizer, model = vbll_model_factory(args.model_name, 1, args.parameterization, args.prior_scale, args.wishart_scale, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    trec_dl_19 = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
    trec_dl_20 = ir_datasets.load("msmarco-passage/trec-dl-2020/judged")

    index = FaissIndex.build(psg_embs[:,0,:])
    print(f'TREC DL 2019: {len(trec_dl_19.queries)} queries')
    qpp(trec_dl_19, model, tokenizer, index, psg_ids, device, unc_method=unc_method)
    print('')

    print(f'TREC DL 2020: {len(trec_dl_20.queries)} queries')
    qpp(trec_dl_20, model, tokenizer, index, psg_ids, device, unc_method=unc_method)


if __name__ == '__main__':
    args = OmegaConf.load('config.yml')
    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    params = SimpleNamespace(**config)
    main(params, args.wandb.run_id, "norm")