from transformers import AutoModel, AutoTokenizer
import vbll
import torch
import torch.nn as nn

_model_registry = {
    "bert-tiny": "google/bert_uncased_L-2_H-128_A-2",
    "bert-mini": "google/bert_uncased_L-4_H-256_A-4",
    "bert-small": "google/bert_uncased_L-4_H-512_A-8",
    "bert-medium": "google/bert_uncased_L-8_H-512_A-8",
    "bert-base": "google/bert_uncased_L-12_H-768_A-12",
    "bert-base-msmarco": "sentence-transformers/msmarco-bert-base-dot-v5",
    "distilbert-base-msmarco-tasb": "sentence-transformers/msmarco-distilbert-base-tas-b",
}


def model_factory(model_name, device):
    if model_name.startswith("bert"):
        retriever_class = BERTRetriever
    
    tokenizer, model = retriever_class.build(get_hf_model_id(model_name), device=device)

    return tokenizer, model


def vbll_model_factory(model_name, device, reg_weight, prior_scale, wishart_scale):
    retriever_class = VBLLRetriever
    tokenizer, model = retriever_class.build(get_hf_model_id(model_name), reg_weight, prior_scale, wishart_scale, device=device)

    return tokenizer, model


def get_hf_model_id(model_name):
    return _model_registry[model_name]


def enable_grad(module):
    for p in module.parameters():
        p.requires_grad = True


def disable_grad(module):
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Retriever(nn.Module):
    def __init__(self, backbone, device="cpu"):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.to(device)

    def forward(self, qry_or_psg):
        return self._encode(qry_or_psg)

    def _encode(self, qry_or_psg):
        model_output = self.backbone(**qry_or_psg, return_dict=True)
        embeddings = self.cls_pooling(model_output, qry_or_psg["attention_mask"])
        return embeddings

    @classmethod
    def build(cls, model_name, device="cpu", **hf_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, **hf_kwargs)
        return tokenizer, cls(backbone, device=device)


class BERTRetriever(Retriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        disable_grad(self.backbone.embeddings)

    def cls_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    

class VBLLRetriever(Retriever):
    def __init__(self, backbone, reg_weight, prior_scale, wishart_scale, device="cpu"):
        super().__init__(backbone, device)
        disable_grad(self.backbone.embeddings)
        dim = self.backbone.config.hidden_size
        self.vbll_layer = vbll.Regression(dim, dim, reg_weight, prior_scale=prior_scale, wishart_scale=wishart_scale)

    def cls_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, qry_or_psg):
        return self._encode(qry_or_psg)
    
    def _encode(self, qry_or_psg):
        model_output = self.backbone(**qry_or_psg, return_dict=True)
        embeddings = self.cls_pooling(model_output, qry_or_psg["attention_mask"])
        output = self.vbll_layer(embeddings)
        return output
    
    @classmethod
    def build(cls, model_name, reg_weight, prior_scale, wishart_scale, device="cpu", **hf_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, **hf_kwargs)
        return tokenizer, cls(backbone, reg_weight, prior_scale, wishart_scale, device=device)