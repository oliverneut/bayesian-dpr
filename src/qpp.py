import torch
from scipy.stats import pearsonr
import torch
import wandb
from omegaconf import OmegaConf
from types import SimpleNamespace
from indexing import FaissIndex
from pytrec_eval import RelevanceEvaluator
from data_loaders import get_qrels

def uncertainty_score(cov):
    return torch.norm(cov)

def score_ranking(ranking):
    pass

def main(args, run_id):
    model_dir = f"{args.output_dir}/{run_id}"
    psg_embs_path = f"{model_dir}/psg_embs.pt"
    psg_ids_path = f"{model_dir}/psg_ids.pt"
    qry_embs_path = f"{model_dir}/qry_embs.pt"
    qry_ids_path = f"{model_dir}/qry_ids.pt"

    psg_embs = torch.load(psg_embs_path)
    psg_ids = torch.load(psg_ids_path)
    qry_embs = torch.load(qry_embs_path)
    qry_ids = torch.load(qry_ids_path)

    index = FaissIndex.build(psg_embs[:,0,:])

    qrels = get_qrels(split="dev")

    uncertainty_scores = []
    ncdg_scores = []
    mrr_scores = []
    
    with torch.no_grad():
        for qry_emb, qry_id in zip(qry_embs, qry_ids):
            mean, cov = qry_emb
            run = {}
            uncertainty = uncertainty_score(cov).item()
            uncertainty_scores.append(uncertainty)

            # scores, indices = index.search(mean, k=10)
            scores, indices = index.search(mean.unsqueeze(0), k=10)
            batch_psg_indices = [psg_ids[idx] for idx in indices[0]]

            run[qry_id] = {}
            for score, psg_id in zip(scores[0], batch_psg_indices):
                run[qry_id][psg_id] = float(score)

            evaluator = RelevanceEvaluator(qrels, {"ndcg", "recip_rank"})
            results = evaluator.evaluate(run)[qry_id]
            ndcg, mrrr = results.values()

            ncdg_scores.append(ndcg)
            mrr_scores.append(mrrr)

    return pearsonr(uncertainty_scores, ncdg_scores), pearsonr(uncertainty_scores, mrr_scores)

if __name__ == '__main__':
    wandb_args = OmegaConf.load('src/utils/config.yml').wandb
    run_id = "79mqroci"
    api = wandb.Api()
    config = api.run(f"{wandb_args.entity}/{wandb_args.project}/{run_id}").config
    args = SimpleNamespace(**config)
    main(args, run_id)