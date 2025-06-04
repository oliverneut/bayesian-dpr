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
from types import SimpleNamespace

logger = logging.getLogger(__name__)

def main(params: SimpleNamespace, data_cfg: DatasetConfig, run_id: str):
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset id: {data_cfg.dataset_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    save_dir = f"output/models/{run_id}"
    model_path = f"{save_dir}/model.pt"

    parameterization = "diagonal"
    if params.knowledge_distillation:
        tokenizer, model = vbll_model_factory(params.model_name, 1, parameterization, params.prior_scale, params.wishart_scale, device)
        method = "vbll"
    else:
        tokenizer, model = model_factory(params.model_name, device)
        method = "dpr"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_queries = get_query_dataloader(data_cfg.get_query_file(split=data_cfg.test_name), batch_size=params.batch_size, shuffle=False)
    corpus = get_corpus_dataloader(data_cfg.get_corpus_file(), batch_size=params.batch_size, shuffle=False)
    qrels = get_qrels(data_cfg.get_qrels_file(split=data_cfg.test_name))

    psg_embs, psg_ids = encode_corpus(corpus, tokenizer, model, device, method=method)
    index = FaissIndex.build(psg_embs)
    evaluator = Evaluator(tokenizer, model, method, device, index=index,
        metrics={"ndcg", "recip_rank"}, psg_ids=psg_ids)
    
    metrics = evaluator.evaluate_retriever(test_queries, qrels, k=params.k)
    ndcg = metrics[f"nDCG@{params.k}"]
    mrr = metrics[f"MRR@{params.k}"]

    logger.info(f"nDCG@{params.k}: {ndcg}")
    logger.info(f"MRR@{params.k}: {mrr}")


if __name__ == '__main__':
    args = OmegaConf.load('config.yml')
    data_cfg = DatasetConfig(args.prepare_data.dataset_id)
    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    params = SimpleNamespace(**config)
    main(params, data_cfg, args.wandb.run_id)