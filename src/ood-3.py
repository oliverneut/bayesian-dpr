from typing import Dict
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
import torch
import logging
from utils.indexing import FaissIndex
from vbll.layers.regression import VBLLReturn
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.data_utils import DatasetConfig
from utils.data_loaders import get_queries, get_qrels
from utils.model_utils import get_model_from_run
from utils.run_utils import RunConfig
from utils.embedding_utils import has_embeddings, load_embeddings


logger = logging.getLogger(__name__)


def prepare_test_queries(test_queries: list, queries: Dict, data_cfg: DatasetConfig, num_samples: int, OOD: bool) -> None:
    """Prepare test queries dataset."""
    qrels = get_qrels(data_cfg.get_qrels_file(split=data_cfg.test_name))
    
    i = 0
    for qid, rels in qrels.items():
        if len(rels) > 0:
            test_queries.append({"query": queries[qid], "OOD": OOD})
            i += 1
        
        if i >= num_samples: break
    
    return test_queries


def infer_query(qry: str, tokenizer,  model):
    qry_enc = tokenizer(qry, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    qry_emb = model(qry_enc)
    return qry_emb


def uncertainty_score(qry_emb, unc_method="norm"):
    cov = qry_emb.covariance.squeeze()

    if unc_method == "norm":
        return torch.sqrt(qry_emb.trace_covariance)
    elif unc_method == "trace":
        return qry_emb.trace_covariance
    elif unc_method == "det":
        return qry_emb.logdet_covariance
    elif unc_method == "entropy":
        d = cov.size(0)
        logdet = qry_emb.logdet_covariance
        return 0.5 * d * torch.log(torch.tensor(2 * torch.pi * torch.e)) + 0.5 * logdet
    else:
        raise ValueError(f"Unknown uncertainty method: {unc_method}")


def calculate_uncertainty_scores(data, tokenizer, model, unc_method="norm"):
    uncertainty_scores = []
    labels = []
    for query_data in tqdm(data, desc="Calculating uncertainty scores"):       
        emb = infer_query(query_data['query'], tokenizer, model)

        uncertainty_scores.append(uncertainty_score(emb.predictive, unc_method).item())
        labels.append(query_data['OOD'])

    return np.array(uncertainty_scores), np.array(labels)


def metrics(uncertainty_scores, labels):
    auc = roc_auc_score(labels, uncertainty_scores)
    logger.info(f"AUROC: {auc}")
    aupr = average_precision_score(labels, uncertainty_scores)
    logger.info(f"AUPR: {aupr}")
    pbs = pointbiserialr(labels, uncertainty_scores)
    logger.info(f"Point Biserial Correlation: {pbs.correlation}, p-value: {pbs.pvalue}")


def calculate_baseline_scores(queries, tokenizer,  model, index, T=50):
    
    msp_scores = []
    entropy_scores = []
    energy_scores = []
    labels = []

    for query_data in tqdm(queries, desc="Calculating baseline scores"):
        qry_emb = infer_query(query_data['query'], tokenizer, model)
        labels.append(query_data['OOD'])
    
        if isinstance(qry_emb, VBLLReturn):
            qry_emb = qry_emb.predictive.mean
        
        scores, _ = index.search(qry_emb, k=10)
        scores = torch.from_numpy(scores[0])
        shifted_scores = scores - scores.max()
        scaled_scores = shifted_scores / T
        probs = torch.softmax(scaled_scores, dim=0)

        msp_score = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        energy = -torch.log(torch.sum(torch.exp(scaled_scores))).item()

        msp_scores.append(msp_score)
        entropy_scores.append(entropy)
        energy_scores.append(energy)

    return np.array(msp_scores), np.array(entropy_scores), np.array(energy_scores), np.array(labels)


def main(run_cfg: RunConfig, embs_dir: str, T: int = 50, rel_mode: str = "dpr"):
    logger.info(f"Run ID: {run_cfg.run_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer, model = get_model_from_run(run_cfg, device)
    data_cfg = DatasetConfig('msmarco')

    if has_embeddings(run_cfg, data_cfg, embs_dir):
        psg_embs, _ = load_embeddings(run_cfg, data_cfg, embs_dir, rel_mode, device)
    else:
        logger.info("No precomputed embeddings found. Please run the eval_retriever script first.")
        return
    
    index = FaissIndex.build(psg_embs)
    msmarco_queries = get_queries(data_cfg.get_queries_file())

    for ood_dataset in ['nq', 'hotpotqa', 'fiqa']:
        ood_cfg = DatasetConfig(ood_dataset)
        ood_queries = get_queries(ood_cfg.get_queries_file())

        msmarco_ood_queries = []
        msmarco_ood_queries = prepare_test_queries(msmarco_ood_queries, msmarco_queries, data_cfg, 1000, OOD=False)
        msmarco_ood_queries = prepare_test_queries(msmarco_ood_queries, ood_queries, ood_cfg, 1000, OOD=True)
        
        unc_method = "norm"
        if run_cfg.vbll:
            uncertainty_scores, labels = calculate_uncertainty_scores(msmarco_ood_queries, tokenizer, model, unc_method=unc_method)
            logger.info(f"Uncertainty scores calculated using method {unc_method}")
            metrics(uncertainty_scores, labels)
        
        logger.info('')
        msp_scores, entropy_scores, energy_scores, labels = calculate_baseline_scores(msmarco_ood_queries, tokenizer, model, index, T)
        logger.info(f"Baseline scores calculated")
        metrics(msp_scores, labels)
        metrics(entropy_scores, labels)
        metrics(energy_scores, labels)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    args = OmegaConf.load('config.yml')

    run_cfg = RunConfig(args)
    logger.info(f"Run ID: {args.wandb.run_id}")
    main(run_cfg, embs_dir=args.eval.embs_dir)