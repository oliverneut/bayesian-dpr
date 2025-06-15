from omegaconf import OmegaConf
import logging
import torch
from utils.model_utils import vbll_model_factory, model_factory
from data_loaders import get_qrels, get_corpus_dataloader, get_query_dataloader
from encoding import encode_corpus
from evaluation import Evaluator
from indexing import FaissIndex
from utils.data_utils import DatasetConfig
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm
import os


logger = logging.getLogger(__name__)


class VBLLEmbeddingDataset(Dataset):
    def __init__(self, qry_embs_path, qry_ids_path):
        self.data = torch.load(qry_embs_path)
        self.ids = torch.load(qry_ids_path)
        self._num_samples = len(self.data)

    def __len__(self):
        return self._num_samples
    
    def __getitem__(self, i):
        return self.data[i][0], self.data[i][1], self.ids[i]


def process_embeddings_kl(psg_embs: DataLoader):
    processed_embs = []
    psg_ids = []
    
    for psg_emb in tqdm(psg_embs, desc="Processsing embeddings into KL format"):
        mean, cov, psg_id = psg_emb[0], psg_emb[1], psg_emb[2]
        doc_prior = torch.sum(torch.log(cov) + (torch.square(mean) / cov), dim=1).unsqueeze(1)
        inv_cov = 1 / cov
        psg_emb = torch.cat([doc_prior, inv_cov, inv_cov, (-2 * mean) * inv_cov], dim=1)

        processed_embs.append(psg_emb.detach().cpu())
        psg_ids += list(psg_id)

    processed_embs = torch.cat(processed_embs, dim=0)
    return processed_embs, psg_ids

def main(model_name: str, vbll: bool, data_cfg: DatasetConfig, run_id: str, eval_mode: str = "dpr"):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset id: {data_cfg.dataset_id}")
    logger.info(f"Evaluation mode: {eval_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    save_dir = f"output/models/{run_id}"
    # save_dir = f"/scratch-shared/tmp.alIhTYBxk7/{run_id}"
    model_path = f"{save_dir}/model.pt"
    embs_path = f"{save_dir}/{data_cfg.dataset_id}"

    if vbll:
        tokenizer, model = vbll_model_factory(model_name, device)
    else:
        tokenizer, model = model_factory(model_name, device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_queries = get_query_dataloader(data_cfg.get_query_file(split=data_cfg.test_name), batch_size=16, shuffle=False)
    qrels = get_qrels(data_cfg.get_qrels_file(split=data_cfg.test_name))
    
    if os.path.exists(f"{embs_path}/psg_embs.pt") and os.path.exists(f"{embs_path}/psg_ids.pt"):
        logger.info("Loading precomputed embeddings and IDs from disk.")
        if eval_mode == "kl":
            logger.info("Processing precomputed corpus embeddings to KL index")
            psg_embs_dataset = VBLLEmbeddingDataset(f"{embs_path}/psg_embs.pt", f"{embs_path}/psg_ids.pt")
            psg_embs_dataloader = DataLoader(psg_embs_dataset, batch_size=16, shuffle=False)
            psg_embs, psg_ids = process_embeddings_kl(psg_embs_dataloader)
        else:
            psg_embs = torch.load(f"{embs_path}/psg_embs.pt", map_location=device)
            psg_ids = torch.load(f"{embs_path}/psg_ids.pt")
            if psg_embs.dim() == 3:
                logger.info("Reshaping embeddings from 3D to 2D.")
                psg_embs = psg_embs[:,0]
    else:
        corpus = get_corpus_dataloader(data_cfg.get_corpus_file(), batch_size=16, shuffle=False)
        psg_embs, psg_ids = encode_corpus(corpus, tokenizer, model, eval_mode=eval_mode, device=device)

    index = FaissIndex.build(psg_embs)
    
    evaluator = Evaluator(tokenizer, model, eval_mode, device, index=index,
        metrics={"ndcg", "recip_rank"}, psg_ids=psg_ids)
    
    metrics = evaluator.evaluate_retriever(test_queries, qrels, k=10)
    
    logger.info(f"nDCG@{10}: {metrics[f"nDCG@{10}"]}")
    logger.info(f"MRR@{10}: {metrics[f"MRR@{10}"]}")


if __name__ == '__main__':
    args = OmegaConf.load('config.yml')
    data_cfg = DatasetConfig(args.eval.dataset_id)
    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    model_id = config['model_name']
    vbll = config['knowledge_distillation']
    main(model_id, vbll, data_cfg, args.wandb.run_id, eval_mode=args.eval.mode)