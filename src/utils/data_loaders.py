import json
import csv
from collections import defaultdict
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import DataLoader, Dataset
import ir_datasets
import random
import torch


def get_queries(query_file: str):
    queries = {}
    with open(query_file, encoding="utf8") as fIn:
        for line in fIn:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")
            
    return queries


def get_corpus(corpus_file: str):
    corpus = {}
    with open(corpus_file, encoding="utf8") as fIn:
        for line in fIn:
            line = json.loads(line)
            corpus[line.get("_id")] = line.get("text")
    
    return corpus


def get_qrels(qrels_file: str):
    if "trec" in qrels_file:
        return ir_datasets.load(qrels_file).qrels_dict()
    
    qrels = defaultdict(dict)
    reader = csv.reader(open(qrels_file, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    for row in reader:
        q_id, p_id, score = row[0], row[1], int(row[2])
        qrels[q_id][p_id] = score
    
    return qrels


def _load_data(data_file):
    if data_file.suffix == ".jsonl":
        data = HuggingFaceDataset.from_json(str(data_file))
    elif "trec" in data_file:
        data = ir_datasets.load(data_file)
    else:
        raise NotImplementedError("Data file with format {} not supported.".format(data_file.split(".")[-1]))
    return data

class EmbeddingDataset(Dataset):
    def __init__(self, qry_embs_path, qry_ids_path):
        self.data = torch.load(qry_embs_path)
        self.ids = torch.load(qry_ids_path)
        self._num_samples = len(self.data)

    def __len__(self):
        return self._num_samples
    
    def __getitem__(self, i):
        return self.data[i], self.ids[i]

class DPRDataset(Dataset):
    def __init__(self, data_file, corpus_file):
        self.data = _load_data(data_file)
        self._num_samples = len(self.data)
        self.corpus = get_corpus(corpus_file)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, i):
        pos_psg_id = random.choice(self.data[i]["pos"])
        neg_psg_id = random.choice(self.data[i]["neg"])
        return (
            self.data[i]["query"],
            self.corpus[str(pos_psg_id)],
            self.corpus[str(neg_psg_id)],
        )
    

class QueryDataset(Dataset):
    def __init__(self, data_file):
        self.data = _load_data(data_file)
        self._num_samples = len(self.data) 
    
    def __len__(self):
        return self._num_samples
    
    def __getitem__(self, i):
        return (self.data[i]["qid"], self.data[i]["query"])

    
class CorpusDataset(Dataset):
    def __init__(self, data_file):
        self.data = _load_data(data_file)
        self._num_samples = len(self.data)
    
    def __len__(self):
        return self._num_samples
    
    def __getitem__(self, i):
        return (self.data[i]["_id"], self.data[i]["text"])


def get_dataloader(data_file, corpus_file, batch_size=32, shuffle=False):
    dataset = DPRDataset(data_file=data_file, corpus_file=corpus_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=True)


def get_query_dataloader(data_file, batch_size=32, shuffle=False):
    dataset = QueryDataset(data_file=data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False)


def get_corpus_dataloader(corpus_file, batch_size=32, shuffle=False):
    dataset = CorpusDataset(data_file=corpus_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False)