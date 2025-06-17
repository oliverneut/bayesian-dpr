import logging

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _ensure_numpy(emb):
    if isinstance(emb, torch.Tensor):
        return emb.detach().cpu().numpy()
    return emb


class FaissIndex:
    def __init__(self, emb_dim):
        index = faiss.IndexFlatIP(emb_dim)
        self.index = index

    def add(self, psg_emb):
        psg_emb = _ensure_numpy(psg_emb)
        self.index.add(psg_emb)

    def search(self, qry_emb, k):
        qry_emb = _ensure_numpy(qry_emb)
        return self.index.search(qry_emb, k)

    def batch_search(self, qry_emb, k, batch_size):
        qry_emb = _ensure_numpy(qry_emb)
        num_queries = qry_emb.shape[0]
        scores = []
        indices = []
        for i in range(0, num_queries, batch_size):
            batch_scores, batch_indices = self.search(qry_emb[i : i + batch_size], k)
            scores.append(batch_scores)
            indices.append(batch_indices)
        scores = np.concatenate(scores, axis=0)
        indices = np.concatenate(indices, axis=0)
        return scores, indices

    @classmethod
    def build(cls, embs):
        index = cls(embs.size(1))
        index.add(embs)
        return index