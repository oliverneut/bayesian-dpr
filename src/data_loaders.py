import json
import csv
from collections import defaultdict
from utils.config import QUERY_FILE, CORPUS_FILE, TRAIN_QRELS_FILE
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import DataLoader, Dataset
import random

def get_queries():
    queries = {}
    with open(QUERY_FILE, encoding="utf8") as fIn:
        for line in fIn:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")
            
    return queries


def get_corpus():
    corpus = {}
    with open(CORPUS_FILE, encoding="utf8") as fIn:
        for line in fIn:
            line = json.loads(line)
            corpus[line.get("_id")] = line.get("text")
    
    return corpus


def get_qrels():
    qrels = defaultdict(dict)
    reader = csv.reader(open(TRAIN_QRELS_FILE, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    for row in reader:
        q_id, p_id, score = row[0], row[1], int(row[2])
        qrels[q_id][p_id] = score
    
    return qrels


def _load_data(data_file):
    if data_file.endswith(".jsonl"):
        data = HuggingFaceDataset.from_json(data_file)
    else:
        raise NotImplementedError("Data file with format {} not supported.".format(data_file.split(".")[-1]))
    return data


class DPRDataset(Dataset):
    def __init__(self, data_file):
        self.data = _load_data(data_file)
        self._num_samples = len(self.data)
        self.corpus = get_corpus()

    def __len__(self):
        return self._num_samples

    def __getitem__(self, i):
        pos_psg = self.data[i]["pos"]
        neg_psg = self.data[i]["neg"]
        pos_psg_id = random.randint(0, len(pos_psg) - 1)
        neg_psg_id = random.randint(0, len(neg_psg) - 1)
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
        return (self.data[i]["pid"], self.data[i]["text"])


def get_dataloader(data_file, batch_size=32, shuffle=False):
    dataset = DPRDataset(data_file=data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=True)


def get_query_dataloader(data_file, batch_size=32, shuffle=False):
    dataset = QueryDataset(data_file=data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False)


def get_corpus_dataloader(corpus_file, batch_size=32, shuffle=False):
    dataset = CorpusDataset(data_file=corpus_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False)