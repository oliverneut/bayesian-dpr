import logging
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from losses import BinaryPassageRetrievalLoss
from indexing import FaissIndex
from evaluation import Evaluator
from encoding import encode_corpus

logger = logging.getLogger(__name__)

def make_lr_scheduler_with_warmup(model, training_data, lr, min_lr, num_epochs, warmup_rate):
    optimizer = Adam(model.parameters(), lr=lr)
    num_training_steps = len(training_data) * num_epochs
    warmup_iters = int(warmup_rate * num_training_steps)
    decay_iters = int((1 - warmup_rate) * num_training_steps)
    decay_factor = min_lr / lr
    warmup = LinearLR(optimizer, start_factor=decay_factor, end_factor=1.0, total_iters=warmup_iters)
    decay = LinearLR(optimizer, start_factor=1.0, end_factor=decay_factor, total_iters=decay_iters)
    scheduler = SequentialLR(optimizer, [warmup, decay], [warmup_iters])
    logger.info("Using linear learning rate scheduling with linear warm-up.")
    logger.info(
        "Total training steps: %d | LR warm-up for %d steps. | LR decay for %d steps.",
        num_training_steps,
        warmup_iters,
        decay_iters,
    )
    return optimizer, scheduler

class DPRTrainer:
    def __init__(self, tokenizer, model, training_data, validation_queries, validation_corpus, qrels, device):
        self.tokenizer = tokenizer
        self.model = model
        self.training_data = training_data
        self.validation_queries = validation_queries
        self.validation_corpus = validation_corpus
        self.qrels = qrels
        self.device = device
        self.loss_func = BinaryPassageRetrievalLoss()

    def train(
        self,
        num_epochs=4,
        lr=5e-6,
        min_lr=5e-8,
        warmup_rate=0.1,
        ckpt_file_name=None,
        k=20,
        max_qry_len=32,
        max_psg_len=256,
        **kwargs
    ):
        optimizer, scheduler = make_lr_scheduler_with_warmup(
            self.model, self.training_data, lr, min_lr, num_epochs, warmup_rate
        )
        if ckpt_file_name is not None:
            max_ndcg_at_k = 0.0
        else:
            max_ndcg_at_k = 1.0
        scaler = torch.amp.GradScaler(enabled=True)
        for epoch in range(1, num_epochs + 1):
            t_start = time.time()
            self.model.train()
            for qry, pos_psg, neg_psg in self.training_data:
                qry_enc = self.tokenizer(
                    qry, padding="max_length", truncation=True, max_length=max_qry_len, return_tensors="pt"
                ).to(self.device)
                pos_enc = self.tokenizer(
                    pos_psg, padding="max_length", truncation=True, max_length=max_psg_len, return_tensors="pt"
                ).to(self.device)
                neg_enc = self.tokenizer(
                    neg_psg, padding="max_length", truncation=True, max_length=max_psg_len, return_tensors="pt"
                ).to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                    optimizer.zero_grad()
                    qry_emb = self.model(qry_enc)
                    pos_emb = self.model(pos_enc)
                    neg_emb = self.model(neg_enc)
                    loss = self.loss_func(qry_emb, pos_emb, neg_emb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                break
            metrics = self._compute_validation_metrics("dpr", k=k)
            ndcg_at_k = metrics["nDCG@" + str(k)]
            mrr_at_k = metrics["MRR@" + str(k)]
            t_end = time.time()
            logger.info("Epoch %d finished in %.2f minutes.", epoch, (t_end - t_start) / 60)
            logger.info("Validation metrics: nDCG@%d=%.3f | MRR@%d=%.3f", k, ndcg_at_k, k, mrr_at_k)
            if ndcg_at_k > max_ndcg_at_k:
                torch.save(self.model.state_dict(), ckpt_file_name)
                max_ndcg_at_k = ndcg_at_k
                logger.info("Model saved in: %s", ckpt_file_name)

    def _compute_validation_metrics(self, method, k=20, **kwargs):
        self.model.eval()
        psg_embs = encode_corpus(self.validation_corpus, self.tokenizer, self.model, self.device, method, **kwargs)
        index = FaissIndex.build(psg_embs)
        evaluator = Evaluator(
            self.tokenizer,
            self.model,
            method,
            self.device,
            index=index,
            metrics={"ndcg", "recip_rank"},
        )
        return evaluator.evaluate_retriever(self.validation_queries, self.qrels, k=k, **kwargs)