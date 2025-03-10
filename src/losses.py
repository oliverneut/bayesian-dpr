import math

import torch
import torch.nn as nn


def dot_product_similarity(qry_reps, psg_reps):
    return qry_reps @ psg_reps.T

class BinaryPassageRetrievalLoss(nn.Module):
    """
    This loss function is defined in: https://aclanthology.org/2021.acl-short.123.pdf
    Implementation adapted from: https://github.com/beir-cellar/beir/blob/main/beir/losses/bpr_loss.py
    """

    def __init__(self, scale=1.0, similarity_fct=None, binary_ranking_loss_margin=2.0, hashnet_gamma=0.1):
        super().__init__()
        self.global_step = 0
        self.scale = scale
        self.similarity_fct = similarity_fct or dot_product_similarity
        self.hashnet_gamma = hashnet_gamma
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=binary_ranking_loss_margin)

    def convert_to_binary(self, x: torch.Tensor) -> torch.Tensor:
        scale_val = math.sqrt(1.0 + self.global_step * self.hashnet_gamma)
        return torch.tanh(x * scale_val)

    def forward(self, query_emb, pos_emb, neg_emb):
        embeddings_b = torch.cat([pos_emb, neg_emb], dim=0)
        scores = self.similarity_fct(query_emb, embeddings_b) * self.scale
        labels = torch.arange(0, query_emb.size(0), dtype=torch.long, device=scores.device)
        dense_loss = self.cross_entropy_loss(scores, labels)
        binary_query = self.convert_to_binary(query_emb)
        binary_b = self.convert_to_binary(embeddings_b)
        binary_scores = torch.matmul(binary_query, binary_b.transpose(0, 1))
        pos_bin_scores = binary_scores[torch.arange(query_emb.size(0)), labels]
        neg_mask = torch.ones_like(binary_scores, dtype=torch.bool)
        neg_mask[torch.arange(query_emb.size(0)), labels] = False
        neg_bin_scores = binary_scores[neg_mask]
        pos_bin_scores_expanded = pos_bin_scores.repeat_interleave((2 * query_emb.size(0) - 1))
        bin_labels = torch.ones_like(pos_bin_scores_expanded, dtype=torch.float32)
        binary_loss = self.margin_ranking_loss(pos_bin_scores_expanded, neg_bin_scores, bin_labels)
        self.global_step += 1
        return dense_loss + binary_loss