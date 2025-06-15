import torch
import logging
from utils.model_utils import vbll_model_factory, model_factory
from data_loaders import get_corpus_dataloader
from utils.data_utils import DatasetConfig
import wandb
from omegaconf import OmegaConf
from types import SimpleNamespace
from vbll.layers.regression import VBLLReturn
from tqdm import tqdm
import os


logger = logging.getLogger(__name__)


def encode_corpus(corpus, tokenizer, encoder, device, max_psg_len=256):
    psg_embs = []
    psg_ids = []

    with torch.no_grad():
        for psg_id, psg in tqdm(corpus, desc="Encoding corpus"):
            psg_enc = tokenizer(psg, padding="max_length", truncation=True, max_length=max_psg_len, return_tensors="pt").to(device)
            
            psg_emb = encoder(psg_enc)
            
            if isinstance(psg_emb, VBLLReturn):
                psg_emb = psg_emb.predictive
                psg_emb = torch.stack([psg_emb.loc, psg_emb.variance], dim=1)
            
            psg_embs.append(psg_emb.detach().cpu())
            psg_ids += list(psg_id)
            
        psg_embs = torch.cat(psg_embs, dim=0)
    return psg_embs, psg_ids


def main(args, data_cfg: DatasetConfig, run_id: str):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(f"Run ID: {run_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_dir = f"output/models/{run_id}"
    model_path = f"{model_dir}/model.pt"
    embs_dir = f"{model_dir}/{data_cfg.dataset_id}"
    os.makedirs(embs_dir, exist_ok=True)

    if args.knowledge_distillation:
        tokenizer, model = vbll_model_factory(args.model_name, device)
    else:
        tokenizer, model = model_factory(args.model_name, device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    corpus = get_corpus_dataloader(data_cfg.get_corpus_file(), batch_size=args.batch_size, shuffle=False)
    psg_embs, psg_ids = encode_corpus(corpus, tokenizer, model, device)

    torch.save(psg_embs, f"{embs_dir}/psg_embs.pt")
    torch.save(psg_ids, f"{embs_dir}/psg_ids.pt")

    read_embs = torch.load(f"{embs_dir}/psg_embs.pt")
    
    # Check if the embeddings are the same
    assert torch.all(torch.eq(psg_embs, read_embs)), "Embeddings do not match!"
    logger.info("Embeddings saved and verified successfully.")


if __name__ == '__main__':
    args = OmegaConf.load('config.yml')
    data_cfg = DatasetConfig(args.eval.dataset_id)
    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    params = SimpleNamespace(**config)
    main(params, data_cfg, args.wandb.run_id)