from typing import Dict
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from sklearn.cluster import KMeans


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


def uncertainty_score(qry_emb, unc_method="norm"):
    cov = qry_emb.covariance.squeeze()

    if unc_method == "max":
        return torch.max(qry_emb.variance, dim=1).values
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


def embed_queries(data, tokenizer, model, device):
    labels = []
    qry_embs = []

    for qry, ood in tqdm(data, desc="Embedding queries"):   
        qry_enc = tokenizer(qry, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)

        qry_emb = model(qry_enc)

        if isinstance(qry_emb, VBLLReturn):
            qry_emb = qry_emb.predictive.mean

        labels += ood.tolist()
        qry_embs += qry_emb.detach().tolist()

    return np.array(qry_embs), np.array(labels)


def calculate_uncertainty_scores(data, tokenizer, model, device, unc_method="norm"):
    uncertainty_scores = []
    labels = []
    for qry, ood in tqdm(data, desc="Calculating uncertainty scores"):   
        qry_enc = tokenizer(qry, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)
        qry_emb = model(qry_enc).predictive

        uncertainty_scores += uncertainty_score(qry_emb, unc_method).tolist()
        labels += ood.tolist()

    return np.array(uncertainty_scores), np.array(labels)


def report_metrics(uncertainty_scores, labels):
    auc = roc_auc_score(labels, uncertainty_scores)
    logger.info(f"AUROC: {auc}")
    aupr = average_precision_score(labels, uncertainty_scores)
    logger.info(f"AUPR: {aupr}")
    pbs = pointbiserialr(labels, uncertainty_scores)
    logger.info(f"Point Biserial Correlation: {pbs.correlation}, p-value: {pbs.pvalue}")


def calculate_baseline_scores(queries, tokenizer,  model, index, device, T):
    
    msp_scores = []
    entropy_scores = []
    energy_scores = []
    labels = []

    for qry, ood in tqdm(queries, desc="Calculating baseline scores"):
        qry_enc = tokenizer(qry, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)
        qry_emb = model(qry_enc)
        labels += ood.tolist()
    
        if isinstance(qry_emb, VBLLReturn):
            qry_emb = qry_emb.predictive.mean
        
        scores, _ = index.search(qry_emb, k=10)
        scores = torch.from_numpy(scores)
        max_scores, _ = scores.max(dim=1)
        shifted_scores = scores - max_scores.unsqueeze(1)
        scaled_scores = shifted_scores / T
        probs = torch.softmax(scaled_scores, dim=1)

        msp, _ = torch.max(probs, dim=1)
        msp_scores += msp.tolist()
        entropy_scores += (-torch.sum(probs * torch.log(probs + 1e-10), dim=1)).tolist()
        energy_scores += (-torch.log(torch.sum(torch.exp(scaled_scores), dim=1))).tolist()

    return np.array(msp_scores), np.array(entropy_scores), np.array(energy_scores), np.array(labels)


class QueryDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self._num_samples = len(self.data)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, i):
        return self.data[i]['query'], self.data[i]['OOD']


def create_eval_dataset(queries, data_cfg: DatasetConfig, ood_dataset: str):
    ood_cfg = DatasetConfig(ood_dataset)
    ood_queries = get_queries(ood_cfg.get_queries_file())
    num_samples = 1000  # Default number of samples for OOD datasets
    if ood_dataset == 'fiqa':
        num_samples = 500
    eval_queries = prepare_test_queries([], queries, data_cfg, num_samples, OOD=False)
    eval_queries = prepare_test_queries(eval_queries, ood_queries, ood_cfg, num_samples, OOD=True)

    query_dataset = QueryDataset(eval_queries)
    return DataLoader(query_dataset, batch_size=16, shuffle=False)


def main(run_cfg: RunConfig, embs_dir: str, T: int = 5, rel_mode: str = "dpr"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer, model = get_model_from_run(run_cfg, device)
    msmarco_cfg = DatasetConfig('msmarco')

    # if has_embeddings(run_cfg, msmarco_cfg, embs_dir):
    #     psg_embs, _ = load_embeddings(run_cfg, msmarco_cfg, embs_dir, rel_mode, device)
    # else:
    #     logger.info("No precomputed embeddings found. Please run the eval_retriever script first.")
    #     return
    
    # index = FaissIndex.build(psg_embs)
    msmarco_queries = get_queries(msmarco_cfg.get_queries_file())

    for ood_dataset in ['nq', 'hotpotqa', 'fiqa']:
        logger.info(f"Processing OOD dataset: {ood_dataset}")
        query_dl = create_eval_dataset(msmarco_queries, msmarco_cfg, ood_dataset)

        if run_cfg.vbll:
            for unc_method in ["max"]:
                uncertainty_scores, labels = calculate_uncertainty_scores(query_dl, tokenizer, model, device, unc_method=unc_method)
                logger.info(f"Uncertainty scores calculated using method {unc_method}")
                report_metrics(uncertainty_scores, labels)
        
        # logger.info('')
        # msp_scores, entropy_scores, energy_scores, labels = calculate_baseline_scores(query_dl, tokenizer, model, index, device, T)
        # logger.info(f"Baseline scores calculated")
        # report_metrics(msp_scores, labels)
        # report_metrics(entropy_scores, labels)
        # report_metrics(energy_scores, labels)

        logger.info("Clustering uncertainty scores")
        qry_embs, labels = embed_queries(query_dl, tokenizer, model, device)
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(qry_embs)
        report_metrics(clusters, labels)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info("OOD-3 experiment")
    args = OmegaConf.load('config.yml')

    run_cfg = RunConfig(args)
    logger.info(f"Run ID: {args.wandb.run_id}")
    main(run_cfg, embs_dir=args.eval.embs_dir)