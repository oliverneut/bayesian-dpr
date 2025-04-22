import json
import logging
import os
import time

import numpy as np
import torch
from pytrec_eval import RelevanceEvaluator
from encoding import encode_query_mean


logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, tokenizer, model, method, device, index=None, metrics=None, psg_ids=None):
        if metrics is None:
            metrics = {"ndcg", "recip_rank"}
        self.tokenizer = tokenizer
        self.model = model
        self.method = method
        self.device = device
        self.index = index
        self.metrics = metrics
        self.psg_ids = psg_ids

    def evaluate_retriever(self, qry_data_loader, qrels, k=20, num_samples=None, max_qry_len=32, run_file=None):
        if run_file is not None and os.path.exists(run_file) and os.path.isfile(run_file):
            logger.info("Loading run from: %s", run_file)
            with open(run_file, "r", encoding="utf-8") as f:
                run = json.loads(f.read())
        else:
            logger.info("Generating run...")
            t_start = time.time()
            run = self._generate_run(qry_data_loader, k=k, num_samples=num_samples, max_qry_len=max_qry_len)
            t_end = time.time()
            logger.info("Run generated in %.2f minutes.", (t_end - t_start) / 60)
            if run_file is not None:
                with open(run_file, "w", encoding="utf-8") as f:
                    json.dump(run, f)
                logger.info("Run saved in: %s", run_file)
        logger.info("Calculating metrics...")
        results = self._calculate_metrics(run, qrels, k=k)
        return results

    def _generate_run(self, qry_data_loader, k=20, num_samples=None, max_qry_len=32):
        # if qry_data_loader.batch_size != 1:
        #     raise ValueError("To generate a run, load the queries with a batch size of 1.")
        run = {}
        with torch.no_grad():
            for qry_id, qry in qry_data_loader:
                qry_enc = self.tokenizer(
                    qry, padding="max_length", truncation=True, max_length=max_qry_len, return_tensors="pt"
                ).to(self.device)
                if self.method == "bret":
                    qry_emb = self.model(qry_enc, num_samples=num_samples)
                    qry_emb = encode_query_mean(qry_emb)
                if self.method == "vbll":
                    qry_emb = self.model(qry_enc).predictive.loc
                else:
                    qry_emb = self.model(qry_enc)
                scores, indices = self.index.search(qry_emb, k)
                psg_indices = [self.psg_ids[idx] for idx in indices[0]]
                
                qid = str(qry_id.item())
                run[qid] = {}
                for score, psg_id in zip(scores[0], psg_indices):
                    run[qid][psg_id] = float(score)
        return run

    def _calculate_metrics(self, run, qrels, k=20):
        evaluator = RelevanceEvaluator(qrels, self.metrics)
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