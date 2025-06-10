from omegaconf import OmegaConf
import logging
import torch
from utils.model_utils import vbll_model_factory, model_factory
from data_loaders import get_qrels, get_corpus_dataloader, get_query_dataloader
from encoding import encode_corpus
from evaluation import Evaluator
from indexing import FaissIndex
from utils.data_utils import DatasetConfig
import wandb

logger = logging.getLogger(__name__)

def main(model_name: str, vbll: bool, data_cfg: DatasetConfig, run_id: str, eval_mode: str = "kl"):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset id: {data_cfg.dataset_id}")
    logger.info(f"Evaluation mode: {eval_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    save_dir = f"output/models/{run_id}"
    model_path = f"{save_dir}/model.pt"

    if vbll:
        tokenizer, model = vbll_model_factory(model_name, device)
    else:
        tokenizer, model = model_factory(model_name, device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_queries = get_query_dataloader(data_cfg.get_query_file(split=data_cfg.test_name), batch_size=16, shuffle=False)
    corpus = get_corpus_dataloader(data_cfg.get_corpus_file(), batch_size=16, shuffle=False)
    qrels = get_qrels(data_cfg.get_qrels_file(split=data_cfg.test_name))

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