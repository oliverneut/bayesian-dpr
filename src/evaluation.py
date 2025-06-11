import json
import logging
import os
import time

import numpy as np
import torch
from pytrec_eval import RelevanceEvaluator
from tqdm import tqdm
from vbll.layers.regression import VBLLReturn

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, tokenizer, model, eval_mode, device, index=None, metrics=None, psg_ids=None):
        if metrics is None:
            metrics = {"ndcg", "recip_rank"}
        self.tokenizer = tokenizer
        self.model = model
        self.eval_mode = eval_mode
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
        """Generate run file for evaluation."""
        run = {}
        with torch.no_grad():
            for qry_ids, queries in tqdm(qry_data_loader, desc="Generating run"):
                qry_enc = self.tokenizer(
                    queries, padding="max_length", truncation=True, max_length=max_qry_len, return_tensors="pt"
                ).to(self.device)
                
                qry_emb = self.model(qry_enc)

                if isinstance(qry_emb, VBLLReturn):
                    qry_emb = qry_emb.predictive
                    if self.eval_mode == "kl":
                        mean, cov = qry_emb.mean, qry_emb.variance
                        ones = torch.ones(mean.size(0), 1).to(self.device)
                        qry_emb = -1 * torch.cat([ones, cov, torch.square(mean), mean], dim=1)
                    else:
                        qry_emb = qry_emb.loc
                
                scores, indices = self.index.search(qry_emb, k)
                
                # Convert batch of indices to passage IDs
                batch_psg_indices = [[self.psg_ids[idx] for idx in batch_indices] for batch_indices in indices]
                
                # Process each query in the batch
                for qry_id, query_scores, query_indices in zip(qry_ids, scores, batch_psg_indices):
                    run[qry_id] = {}
                    for score, psg_id in zip(query_scores, query_indices):
                        run[qry_id][psg_id] = float(score)
                        
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