import torch
from scipy.stats import pearsonr, kendalltau
import torch
import wandb
import numpy as np
from omegaconf import OmegaConf
from indexing import FaissIndex
from utils.model_utils import vbll_model_factory
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


class PredictorCalculator:
    def __init__(self, data, dataset_name):
        self.qpp_scores = defaultdict(dict)
        self.scores = defaultdict(dict)
        self.data = data
        self.dataset_name = dataset_name
        self.model = None
        self.tokenizer = None
        self.psg_embs = None
        self.index_reader = LuceneIndexReader.from_prebuilt_index('msmarco-v1-passage')
        self.query = self.get_query_dict()
        self.qtoken_set = self.get_qtoken_set()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def set_model(self, model_dir, model_name):
        model_path = f"{model_dir}/model.pt"
        self.tokenizer, self.model = vbll_model_factory(model_name, self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()


    def calculate_uncertainty_scores(self, model_name, run_id):
        model_dir = f"output/models/{run_id}"
        self.set_model(model_dir, model_name)
        psg_embs = torch.load(f"{model_dir}/psg_embs.pt")
        psg_ids = torch.load(f"{model_dir}/psg_ids.pt")

        index = FaissIndex.build(psg_embs[:,0,:])
        run = defaultdict(dict)

        with torch.no_grad():
            for qry_id, qry in self.data.queries:
                qry_enc = self.tokenizer(qry, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(self.device)
                qry_emb = self.model(qry_enc).predictive

                for unc_method in UNC_METHODS:
                    self.qpp_scores[qry_id][unc_method] = uncertainty_score(qry_emb, unc_method).item()

                scores, indices = index.search(qry_emb.loc, k=10)
                psg_indices = [psg_ids[idx] for idx in indices[0]]
                for score, psg_id in zip(scores[0], psg_indices):
                    run[qry_id][psg_id] = float(score)
            
            evaluator = RelevanceEvaluator(self.data.qrels_dict(), {"ndcg", "recip_rank"})
            results = evaluator.evaluate(run)

            for qry_id, metrics in results.items():
                self.scores[qry_id]['ndcg'] = metrics['ndcg']
                self.scores[qry_id]['mrr'] = metrics['recip_rank']

            ndcg_scores = [self.scores[qry_id]['ndcg'] for qry_id in self.scores]
            mrr_scores = [self.scores[qry_id]['mrr'] for qry_id in self.scores]
            logger.info(f"nDCG: {np.mean(ndcg_scores)}")
            logger.info(f"MRR: {np.mean(mrr_scores)}")


    def calculate_baseline_scores(self):
        qtoken2var, qtoken2std, qtoken2did = self.get_token_stats()

        pred_performance = {}

        for qry_id, qry in tqdm(self.query.items(), total=len(self.query)):
            if qry_id not in self.data.qrels_dict():
                continue

            pred_performance[qry_id] = {}
            qtokens = self.index_reader.analyze(qry)

            self.qpp_scores[qry_id]["QS"] = QS(qtokens, self.index_reader, qtoken2did)

            pmi_avg, pmi_max, pmi_sum = avg_max_sum_PMI(qtokens, self.index_reader, qtoken2did)
            self.qpp_scores[qry_id]["PMI_avg"] = pmi_avg
            self.qpp_scores[qry_id]["PMI_max"] = pmi_max
            self.qpp_scores[qry_id]["PMI_sum"] = pmi_sum

            self.qpp_scores[qry_id]["ql"] = len(qtokens)

            var_avg, var_max, var_sum = avg_max_sum_VAR(qtokens, qtoken2var)
            self.qpp_scores[qry_id]["var_avg"] = var_avg
            self.qpp_scores[qry_id]["var_max"] = var_max
            self.qpp_scores[qry_id]["var_sum"] = var_sum

            std_avg, std_max, std_sum = avg_max_sum_VAR(qtokens, qtoken2std)
            self.qpp_scores[qry_id]["std_avg"] = std_avg
            self.qpp_scores[qry_id]["std_max"] = std_max
            self.qpp_scores[qry_id]["std_sum"] = std_sum

            idf_avg, idf_max, idf_sum, idf_std = avg_max_sum_std_IDF(qtokens, self.index_reader)
            self.qpp_scores[qry_id]["idf_avg"] = idf_avg
            self.qpp_scores[qry_id]["idf_max"] = idf_max
            self.qpp_scores[qry_id]["idf_sum"] = idf_sum

            scq_avg, scq_max, scq_sum = avg_max_sum_SCQ(qtokens, self.index_reader)
            self.qpp_scores[qry_id]["scq_avg"] = scq_avg
            self.qpp_scores[qry_id]["scq_max"] = scq_max
            self.qpp_scores[qry_id]["scq_sum"] = scq_sum

    
    def get_query_dict(self):
        query = {}
        for qry_id, qry in self.data.queries_iter():
            query[qry_id] = qry
        return query
    

    def get_qtoken_set(self):
        qtoken_set = set()

        for _, qry in self.data.queries_iter():
            qtokens = self.index_reader.analyze(qry)
            for qtoken in qtokens:
                qtoken_set.add(qtoken)
        
        return qtoken_set
    

    def token_stats_exists(self):
        qtoken2var = os.path.exists(f"output/qpp/{self.dataset_name}/qtoken2var.json")
        qtoken2std = os.path.exists(f"output/qpp/{self.dataset_name}/qtoken2std.json")
        qtoken2did = os.path.exists(f"output/qpp/{self.dataset_name}/qtoken2did.json")
        return qtoken2var and qtoken2std and qtoken2did
    

    def load_token_stats(self):
        qtoken2var = load_json(f"output/qpp/{self.dataset_name}", "qtoken2var.json")
        qtoken2std = load_json(f"output/qpp/{self.dataset_name}", "qtoken2std.json")
        qtoken2did = load_json(f"output/qpp/{self.dataset_name}", "qtoken2did.json")
        return qtoken2var, qtoken2std, qtoken2did
    

    def save_token_stats(self, qtoken2var, qtoken2std, qtoken2did):
        if not os.path.exists(f"output/qpp/{self.dataset_name}"):
            os.makedirs(f"output/qpp/{self.dataset_name}")

        with open(f"output/qpp/{self.dataset_name}/qtoken2var.json", 'w', encoding='utf-8') as f:
            json.dump(qtoken2var, f, ensure_ascii=False, indent=4)

        with open(f"output/qpp/{self.dataset_name}/qtoken2std.json", 'w', encoding='utf-8') as f:
            json.dump(qtoken2std, f, ensure_ascii=False, indent=4)

        with open(f"output/qpp/{self.dataset_name}/qtoken2did.json", 'w', encoding='utf-8') as f:
            json.dump(qtoken2did, f, ensure_ascii=False, indent=4)


    def get_token_stats(self):
        if self.token_stats_exists():
            logger.info("Loading precomputed token stats.")
            return self.load_token_stats()

        qtoken2var, qtoken2std, qtoken2did = {}, {}, {}

        for qtoken in tqdm(self.qtoken_set):
            qtoken2var[qtoken], qtoken2std[qtoken] = VAR(qtoken, self.index_reader)
            qtoken2did[qtoken] = t2did(qtoken, self.index_reader)

        self.save_token_stats(qtoken2var, qtoken2std, qtoken2did)

        return qtoken2var, qtoken2std, qtoken2did


def calculate_qpp_scores(data, dataset_name, model_name: str, run_id: str, device: torch.device):
    predictor_calc = PredictorCalculator(data, dataset_name)

    predictor_calc.calculate_uncertainty_scores(model_name, run_id)
    
    if False:
        predictor_calc.calculate_baseline_scores()

    return predictor_calc.scores, predictor_calc.qpp_scores


def calculate_correlations(scores, qpp_scores):
    qpp_scores_predictors = qpp_scores[list(qpp_scores.keys())[0]].keys()

    ndcg_scores = [scores[qry_id]['ndcg'] for qry_id in scores]
    mrr_scores = [scores[qry_id]['mrr'] for qry_id in scores]

    for predictor in qpp_scores_predictors:
        predictor_scores = [qpp_scores[qry_id][predictor] for qry_id in qpp_scores]
        
        logger.info(f"Predictor: {predictor}")
        corr, p_val = pearsonr(predictor_scores, ndcg_scores)
        logger.info(f"nDCG Pearson Correlation: {corr} p_value: {p_val}")
        corr, p_val = pearsonr(predictor_scores, mrr_scores)
        logger.info(f"MRR Pearson Correlation: {corr} p_value: {p_val}")
        corr, p_val = kendalltau(predictor_scores, ndcg_scores)
        logger.info(f"nDCG Kendall Tau: {corr} p_value: {p_val}")
        corr, p_val = kendalltau(predictor_scores, mrr_scores)
        logger.info(f"MRR Kendall Tau: {corr} p_value: {p_val}")
        logger.info('------------------------------------------------------')


def main(model_name: str, run_id: str):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(f"Run ID: {run_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_name = "trec-dl-2019"
    trec_dl_19 = ir_datasets.load(f"msmarco-passage/{dataset_name}/judged")
    scores_19, qpp_scores_19 = calculate_qpp_scores(trec_dl_19, dataset_name, model_name, run_id, device)

    logger.info(f"Scores for {dataset_name}")
    calculate_correlations(scores_19, qpp_scores_19)
    logger.info('')

    dataset_name = "trec-dl-2020"
    trec_dl_20 = ir_datasets.load(f"msmarco-passage/{dataset_name}/judged")
    scores_20, qpp_scores_20 = calculate_qpp_scores(trec_dl_20, dataset_name, model_name, run_id, device)

    logger.info(f"Scores for {dataset_name}")
    calculate_correlations(scores_20, qpp_scores_20)


if __name__ == '__main__':
    args = OmegaConf.load('config.yml')
    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    main(config['model_name'], args.wandb.run_id)