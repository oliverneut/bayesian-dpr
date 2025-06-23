import torch
from scipy.stats import pearsonr, kendalltau
import torch
import wandb
import numpy as np
from omegaconf import OmegaConf
from utils.indexing import FaissIndex
from utils.model_utils import get_model_from_run
from utils.run_utils import RunConfig
from utils.embedding_utils import has_embeddings, load_embeddings
from utils.data_utils import DatasetConfig
from torch.utils.data import Dataset
from pytrec_eval import RelevanceEvaluator
from collections import defaultdict
import logging
import ir_datasets
from pyserini.index.lucene import LuceneIndexReader
import numpy as np
import ir_datasets
from tqdm import tqdm
from collections import Counter
import os
import json

logger = logging.getLogger(__name__)


UNC_METHODS = ["norm", "trace", "det", "entropy"]


def IDF(term, index_reader):
    df, cf = index_reader.get_term_counts(term, analyzer=None)

    if df == 0:
        return 0.0
    else:
        return np.log2(index_reader.stats()['documents'] / df)


def SCQ(term, index_reader):
    df, cf = index_reader.get_term_counts(term, analyzer=None)

    if cf == 0:
        return 0.0
    else:
        part_A = 1 + np.log2(cf)
        part_B = IDF(term, index_reader)

    return part_A * part_B


def avg_max_sum_std_IDF(qtokens, index_reader):
    v = []
    for t in qtokens:
        v.append(IDF(t, index_reader))
    return [np.mean(v), max(v), sum(v), np.std(v)]


def avg_max_sum_SCQ(qtokens, index_reader):
    scq = []
    for t in qtokens:
        scq.append(SCQ(t, index_reader))
    return [np.mean(scq), max(scq), sum(scq)]


def ictf(term, index_reader):
    df, cf = index_reader.get_term_counts(term, analyzer=None)
    if cf == 0:
        return 0.0
    else:
        return np.log2(index_reader.stats()['total_terms'] / cf)


def avgICTF(qtokens, index_reader):
    v = []
    for t in qtokens:
        v.append(ictf(t, index_reader))
    return np.mean(v)


def SCS_1(qtokens, index_reader):
    # simplified version
    part_A = np.log2(1 / len(qtokens))
    part_B = avgICTF(qtokens, index_reader)

    return part_A + part_B


def SCS_2(qtokens, index_reader):
    # real version
    v = []
    qtf = Counter(qtokens)
    ql = len(qtokens)

    for t in qtokens:
        pml = qtf[t] / ql
        df, cf = index_reader.get_term_counts(t, analyzer=None)
        pcoll = cf / index_reader.stats()['total_terms']

        if pcoll == 0:
            v.append(0.0)
        else:
            v.append(pml * np.log2(pml / pcoll))

    return sum(v)


def QS(qtokens, index_reader, qtoken2did):
    q2did_set = set()
    for t in qtokens:
        q2did_set = q2did_set.union(set(qtoken2did[t]))

    n_Q = len(q2did_set)
    N = index_reader.stats()['documents']

    return -np.log2(n_Q / N)


def VAR(t, index_reader):
    # one query token, multiple docs containing it
    postings_list = index_reader.get_postings_list(t, analyzer=None)

    if postings_list == None:
        return 0.0, 0.0
    else:
        tf_array = np.array([posting.tf for posting in postings_list])
        tf_idf_array = np.log2(1 + tf_array) * IDF(t, index_reader)

        return np.var(tf_idf_array), np.std(tf_idf_array)


def avg_max_sum_VAR(qtokens, qtoken2x):
    v = []
    for t in qtokens:
        v.append(qtoken2x[t])

    return [np.mean(v), max(v), sum(v)]


def t2did(t, index_reader):
    postings_list = index_reader.get_postings_list(t, analyzer=None)

    if postings_list == None:
        return []
    else:
        return [posting.docid for posting in postings_list]


def PMI(t_i, t_j, index_reader, qtoken2did):
    # We follow the implementation of the paper: https://dl.acm.org/doi/abs/10.1145/1645953.1646114

    titj_doc_count = len(set(qtoken2did[t_i]).intersection(set(qtoken2did[t_j])))
    ti_doc_count = len(qtoken2did[t_i])
    tj_doc_count = len(qtoken2did[t_j])

    if titj_doc_count > 0:
        part_A = titj_doc_count / index_reader.stats()['documents']
        part_B = (ti_doc_count / index_reader.stats()['documents']) * (tj_doc_count / index_reader.stats()['documents'])

        return np.log2(part_A / part_B)
    else:
        return 0.0


def avg_max_sum_PMI(qtokens, index_reader, qtoken2did):
    pair = []
    pair_num = 0

    if len(qtokens) == 0:
        return [0.0, 0.0, 0.0]
    else:
        for i in range(0, len(qtokens)):
            for j in range(i + 1, len(qtokens)):
                pair_num += 1
                pair.append(PMI(qtokens[i], qtokens[j], index_reader, qtoken2did))

        assert len(pair) == pair_num

        if pair_num == 0:
            return [0.0, 0.0, 0.0]

        return [np.mean(pair), max(pair), sum(pair)]


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


def load_json(file_path: str, file_name: str):
    with open(f'{file_path}/{file_name}', 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_correlations(ndcg_scores, mrr_scores, qpp_scores):
    qpp_scores_predictors = qpp_scores[list(qpp_scores.keys())[0]].keys()

    for predictor in qpp_scores_predictors:
        predictor_scores = [qpp_scores[qry_id][predictor] for qry_id in qpp_scores]

        res = []
        logger.info(f"Predictor: {predictor}")
        corr, p_val = pearsonr(predictor_scores, ndcg_scores)
        res.append(corr)
        logger.info(f"nDCG Pearson Correlation: {corr} p_value: {p_val}")
        corr, p_val = pearsonr(predictor_scores, mrr_scores)
        res.append(corr)
        logger.info(f"MRR Pearson Correlation: {corr} p_value: {p_val}")
        corr, p_val = kendalltau(predictor_scores, ndcg_scores)
        res.append(corr)
        logger.info(f"nDCG Kendall Tau: {corr} p_value: {p_val}")
        corr, p_val = kendalltau(predictor_scores, mrr_scores)
        res.append(corr)
        logger.info(f"MRR Kendall Tau: {corr} p_value: {p_val}")
        logger.info('\t'.join(map(str, res)))
        logger.info('------------------------------------------------------')



def get_query_dict(data):
    query = {}
    for qry_id, qry in data.queries_iter():
        query[qry_id] = qry
    return query


def get_qtoken_set(data, index_reader):
    qtoken_set = set()

    for _, qry in data.queries_iter():
        qtokens = index_reader.analyze(qry)
        for qtoken in qtokens:
            qtoken_set.add(qtoken)
    
    return qtoken_set


def token_stats_exists(dataset_name: str):
    qtoken2var = os.path.exists(f"output/qpp/{dataset_name}/qtoken2var.json")
    qtoken2std = os.path.exists(f"output/qpp/{dataset_name}/qtoken2std.json")
    qtoken2did = os.path.exists(f"output/qpp/{dataset_name}/qtoken2did.json")
    return qtoken2var and qtoken2std and qtoken2did


def load_token_stats(dataset_name: str):
    qtoken2var = load_json(f"output/qpp/{dataset_name}", "qtoken2var.json")
    qtoken2std = load_json(f"output/qpp/{dataset_name}", "qtoken2std.json")
    qtoken2did = load_json(f"output/qpp/{dataset_name}", "qtoken2did.json")
    return qtoken2var, qtoken2std, qtoken2did


def save_token_stats(qtoken2var, qtoken2std, qtoken2did, dataset_name: str):
    if not os.path.exists(f"output/qpp/{dataset_name}"):
        os.makedirs(f"output/qpp/{dataset_name}")

    with open(f"output/qpp/{dataset_name}/qtoken2var.json", 'w', encoding='utf-8') as f:
        json.dump(qtoken2var, f, ensure_ascii=False, indent=4)

    with open(f"output/qpp/{dataset_name}/qtoken2std.json", 'w', encoding='utf-8') as f:
        json.dump(qtoken2std, f, ensure_ascii=False, indent=4)

    with open(f"output/qpp/{dataset_name}/qtoken2did.json", 'w', encoding='utf-8') as f:
        json.dump(qtoken2did, f, ensure_ascii=False, indent=4)


def get_token_stats(data, index_reader: LuceneIndexReader, dataset_name: str):
    qtoken2var, qtoken2std, qtoken2did = {}, {}, {}
    
    qtoken_set = get_qtoken_set(data, index_reader)

    for qtoken in tqdm(qtoken_set):
        qtoken2var[qtoken], qtoken2std[qtoken] = VAR(qtoken, index_reader)
        qtoken2did[qtoken] = t2did(qtoken, index_reader)

    save_token_stats(qtoken2var, qtoken2std, qtoken2did, dataset_name)

    return qtoken2var, qtoken2std, qtoken2did


def calculate_baseline_scores(qpp_scores: dict, data: Dataset, dataset_name):
    index_reader = LuceneIndexReader.from_prebuilt_index('msmarco-v1-passage')

    if token_stats_exists(dataset_name):
        logger.info("Loading precomputed token stats.")
        qtoken2var, qtoken2std, qtoken2did = load_token_stats(dataset_name)
    else:
        logger.info("Calculating token stats.")
        qtoken2var, qtoken2std, qtoken2did = get_token_stats(data, index_reader, dataset_name)

    pred_performance = {}

    query_dict = get_query_dict(data)

    for qry_id, qry in tqdm(query_dict.items(), total=len(query_dict)):
        if qry_id not in data.qrels_dict():
            continue
        
        pred_performance[qry_id] = {}
        qtokens = index_reader.analyze(qry)

        qpp_scores[qry_id]["QS"] = QS(qtokens, index_reader, qtoken2did)

        pmi_avg, pmi_max, pmi_sum = avg_max_sum_PMI(qtokens, index_reader, qtoken2did)
        qpp_scores[qry_id]["PMI_avg"] = pmi_avg
        qpp_scores[qry_id]["PMI_max"] = pmi_max
        qpp_scores[qry_id]["PMI_sum"] = pmi_sum

        qpp_scores[qry_id]["ql"] = len(qtokens)

        var_avg, var_max, var_sum = avg_max_sum_VAR(qtokens, qtoken2var)
        qpp_scores[qry_id]["var_avg"] = var_avg
        qpp_scores[qry_id]["var_max"] = var_max
        qpp_scores[qry_id]["var_sum"] = var_sum

        std_avg, std_max, std_sum = avg_max_sum_VAR(qtokens, qtoken2std)
        qpp_scores[qry_id]["std_avg"] = std_avg
        qpp_scores[qry_id]["std_max"] = std_max
        qpp_scores[qry_id]["std_sum"] = std_sum

        idf_avg, idf_max, idf_sum, idf_std = avg_max_sum_std_IDF(qtokens, index_reader)
        qpp_scores[qry_id]["idf_avg"] = idf_avg
        qpp_scores[qry_id]["idf_max"] = idf_max
        qpp_scores[qry_id]["idf_sum"] = idf_sum

        scq_avg, scq_max, scq_sum = avg_max_sum_SCQ(qtokens, index_reader)
        qpp_scores[qry_id]["scq_avg"] = scq_avg
        qpp_scores[qry_id]["scq_max"] = scq_max
        qpp_scores[qry_id]["scq_sum"] = scq_sum

    return qpp_scores


def calculate_uncertainty_scores(data, tokenizer, model, index, psg_ids, device, noise: bool):
    qpp_scores = defaultdict(dict)
    run = defaultdict(dict)

    with torch.no_grad():
        for qry_id, qry in data.queries:
            qry_enc = tokenizer(qry, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)
            qry_emb = model.forward(qry_enc, noise=noise).predictive

            for unc_method in UNC_METHODS:
                qpp_scores[qry_id][unc_method] = uncertainty_score(qry_emb, unc_method).item()

            scores, indices = index.search(qry_emb.loc, k=10)
            psg_indices = [psg_ids[idx] for idx in indices[0]]
            for score, psg_id in zip(scores[0], psg_indices):
                run[qry_id][psg_id] = float(score)
        
    evaluator = RelevanceEvaluator(data.qrels_dict(), {"ndcg", "ndcg_cut_10", "recip_rank"})
    results = evaluator.evaluate(run)

    for qry_id, metrics in results.items():
        qpp_scores[qry_id]['ndcg_cut_10'] = metrics['ndcg_cut_10']
        qpp_scores[qry_id]['mrr'] = metrics['recip_rank']

    ndcg_scores = [qpp_scores[qry_id]['ndcg_cut_10'] for qry_id in qpp_scores]
    mrr_scores = [qpp_scores[qry_id]['mrr'] for qry_id in qpp_scores]
    logger.info(f"ndcg_cut_10: {np.mean(ndcg_scores)}")
    logger.info(f"MRR: {np.mean(mrr_scores)}")
    
    return ndcg_scores, mrr_scores, qpp_scores


def main(run_cfg: RunConfig, embs_dir: str, noise: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer, model = get_model_from_run(run_cfg, device)

    data_cfg = DatasetConfig('msmarco')

    if has_embeddings(run_cfg, data_cfg, embs_dir):
        psg_embs, psg_ids = load_embeddings(run_cfg, data_cfg, embs_dir, rel_mode="dpr", device=device)
    else:
        logger.info("No precomputed embeddings found. Please run the eval_retriever script first.")
        return

    index = FaissIndex.build(psg_embs)

    for dataset_name in ["trec-dl-2019", "trec-dl-2020"]:
        trec_dl = ir_datasets.load(f"msmarco-passage/{dataset_name}/judged")
        ndcg_scores, mrr_scores, qpp_scores = calculate_uncertainty_scores(trec_dl, tokenizer, model, index, psg_ids, device, noise)
        qpp_scores = calculate_baseline_scores(qpp_scores, trec_dl, dataset_name)

        logger.info(f"Calculating correlations for {dataset_name}")
        calculate_correlations(ndcg_scores, mrr_scores, qpp_scores)
        logger.info(f"Finished processing {dataset_name}")


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info("QPP experiment")
    args = OmegaConf.load('config.yml')
    noise = True
    logger.info(f'Noise: {noise}')

    logger.info(f"Run ID: {args.wandb.run_id}")

    run_cfg = RunConfig(args)

    main(run_cfg, args.eval.embs_dir, noise)