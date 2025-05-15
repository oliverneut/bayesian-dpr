import torch
import logging
from utils.model_utils import vbll_model_factory, model_factory
from data_loaders import get_corpus_dataloader, get_query_dataloader
from utils.data_utils import get_query_file
import wandb
from omegaconf import OmegaConf
from types import SimpleNamespace
from tqdm import tqdm

logger = logging.getLogger(__name__)

def encode_corpus(corpus, tokenizer, encoder, device, method, max_psg_len=256):
    psg_embs = []
    psg_ids = []

    with torch.no_grad():
        for psg_id, psg in tqdm(corpus, desc="Encoding corpus"):
            psg_enc = tokenizer(psg, padding="max_length", truncation=True, max_length=max_psg_len, return_tensors="pt").to(device)
            if method == "vbll":
                psg_emb = encoder(psg_enc).predictive
                psg_emb = torch.stack([psg_emb.loc, psg_emb.scale], dim=1)
            else:
                psg_emb = encoder(psg_enc)
            
            psg_embs.append(psg_emb.detach().cpu())
            psg_ids += list(psg_id)
            
        psg_embs = torch.cat(psg_embs, dim=0)
    return psg_embs, psg_ids


def encode_queries(queries, tokenizer, encoder, device, method, max_qry_len=32):
    qry_embs = []
    qry_ids = []

    with torch.no_grad():
        for qry_id, qry in tqdm(queries, desc="Encoding queries"):
            qry_enc = tokenizer(qry, padding="max_length", truncation=True, max_length=max_qry_len, return_tensors="pt").to(device)
            if method == "vbll":
                qry_emb = encoder(qry_enc).predictive
                qry_emb = torch.stack([qry_emb.loc, qry_emb.scale], dim=1)
            else:
                qry_emb = encoder(qry_enc)
            
            qry_embs.append(qry_emb.detach().cpu())
            qry_ids += list(qry_id)

    qry_embs = torch.cat(qry_embs, dim=0)
    return qry_embs, qry_ids


def main(args, run_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_dir = f"output/models/{run_id}"
    model_path = f"{model_dir}/model.pt"

    if args.knowledge_distillation:
        tokenizer, model = vbll_model_factory(args.model_name, 1, args.parameterization, args.prior_scale, args.wishart_scale, device)
        method = "vbll"
    else:
        tokenizer, model = model_factory(args.model_name, device)
        method = "dpr"
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    corpus = get_corpus_dataloader("data/msmarco/corpus.jsonl",  batch_size=args.batch_size, shuffle=False)
    psg_embs, psg_ids = encode_corpus(corpus, tokenizer, model, device, method=method)

    qry_data_loader = get_query_dataloader(get_query_file(split="dev"),  batch_size=args.batch_size, shuffle=False)
    qry_embs, qry_ids = encode_queries(qry_data_loader, tokenizer, model, device, method=method)

    torch.save(psg_embs, f"{model_dir}/psg_embs.pt")
    torch.save(psg_ids, f"{model_dir}/psg_ids.pt")
    torch.save(qry_embs, f"{model_dir}/qry_embs.pt")
    torch.save(qry_ids, f"{model_dir}/qry_ids.pt")

    read_embs = torch.load(f"{model_dir}/psg_embs.pt")
    
    # Check if the embeddings are the same
    assert torch.all(torch.eq(psg_embs, read_embs)), "Embeddings do not match!"
    logger.info("Embeddings saved and verified successfully.")

if __name__ == '__main__':
    wandb_args = OmegaConf.load('src/utils/config.yml').wandb
    run_id = "10nfecme"
    api = wandb.Api()
    config = api.run(f"{wandb_args.entity}/{wandb_args.project}/{run_id}").config
    args = SimpleNamespace(**config)
    main(args, run_id)