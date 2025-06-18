from transformers import AutoModel, AutoTokenizer
from utils.data_utils import DatasetConfig
from utils.run_utils import RunConfig
from torch.utils.data import DataLoader, Dataset
from utils.data_loaders import get_corpus_dataloader
from utils.encoding import encode_corpus
from tqdm import tqdm
import logging
import torch
import os


logger = logging.getLogger(__name__)


class VBLLEmbeddingDataset(Dataset):
    def __init__(self, embs):
        self.embs = embs
        self._num_samples = len(self.embs)

    def __len__(self):
        return self._num_samples
    
    def __getitem__(self, i):
        return self.embs[i][0], self.embs[i][1]


def process_embeddings_kl(psg_embs: DataLoader):
    processed_embs = []
    
    for mean, cov in tqdm(psg_embs, desc="Processsing embeddings into KL format"):
        doc_prior = torch.sum(torch.log(cov) + (torch.square(mean) / cov), dim=1).unsqueeze(1)
        inv_cov = 1 / cov
        psg_emb = torch.cat([doc_prior, inv_cov, inv_cov, (-2 * mean) * inv_cov], dim=1)

        processed_embs.append(psg_emb.detach().cpu())

    logger.info("Concatenating the processed embs")
    processed_embs = torch.cat(processed_embs, dim=0)
    return processed_embs


def has_embeddings(run_cfg: RunConfig, data_cfg: DatasetConfig, embs_dir: str = "output/models"):
    embs_path = get_base_path(run_cfg, data_cfg, embs_dir)
    return os.path.exists(f"{embs_path}/psg_embs.pt") and os.path.exists(f"{embs_path}/psg_ids.pt")


def load_embeddings(run_cfg: RunConfig, data_cfg: DatasetConfig, embs_dir: str, rel_mode: str, device: torch.device):
    logger.info("Loading precomputed embeddings and IDs from disk.")
    embs_path = get_base_path(run_cfg, data_cfg, embs_dir)

    psg_embs = torch.load(f"{embs_path}/psg_embs.pt", map_location=device)
    psg_ids = torch.load(f"{embs_path}/psg_ids.pt")

    if psg_embs.dim() == 3:
        if rel_mode == "dpr":
            psg_embs = psg_embs[:, 0]
        else:
            psg_embs_dataset = VBLLEmbeddingDataset(psg_embs)
            psg_embs_dataloader = DataLoader(psg_embs_dataset, batch_size=16, shuffle=False)
            psg_embs = process_embeddings_kl(psg_embs_dataloader)

    return psg_embs, psg_ids

        
def get_embeddings(run_cfg: RunConfig,
                   data_cfg: DatasetConfig,
                   tokenizer: AutoTokenizer,
                   model: AutoModel,
                   embs_dir: str,
                   rel_mode: str,
                   device: torch.device):
    logger.info("Computing embeddings for corpus.")

    corpus = get_corpus_dataloader(data_cfg.get_corpus_file(), batch_size=16, shuffle=False)
    psg_embs, psg_ids = encode_corpus(corpus, tokenizer, model, device=device)
    
    base_path = get_base_path(run_cfg, data_cfg, embs_dir)
    make_embs_dir(base_path)
    torch.save(psg_embs, f"{base_path}/psg_embs.pt")
    torch.save(psg_ids, f"{base_path}/psg_ids.pt")

    if psg_embs.dim() == 3:
        if rel_mode == "dpr":
            psg_embs = psg_embs[:, 0]
        else:
            psg_embs_dataset = VBLLEmbeddingDataset(psg_embs, psg_ids)
            psg_embs_dataloader = DataLoader(psg_embs_dataset, batch_size=16, shuffle=False)
            psg_embs = process_embeddings_kl(psg_embs_dataloader)

    return psg_embs, psg_ids


def make_embs_dir(base_path: str):
    os.makedirs(base_path, exist_ok=True)


def get_base_path(run_cfg: RunConfig, data_cfg: DatasetConfig, embs_dir: str = "output/models"):
    if data_cfg.dataset_id in {"trec-dl-2019", "trec-dl-2020", "dl-typo"}:
        return f"{embs_dir}/{run_cfg.run_id}/msmarco"
    return f"{embs_dir}/{run_cfg.run_id}/{data_cfg.dataset_id}"