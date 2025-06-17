from omegaconf import OmegaConf
import logging
import torch
from utils.model_utils import get_model_from_run
from utils.data_loaders import get_qrels, get_query_dataloader
from utils.embedding_utils import has_embeddings, load_embeddings, get_embeddings
from utils.evaluation import Evaluator
from utils.indexing import FaissIndex
from utils.data_utils import DatasetConfig
from utils.run_utils import RunConfig


logger = logging.getLogger(__name__)


def main(run_cfg: RunConfig, data_cfg: DatasetConfig, rel_mode: str = "dpr", embs_dir: str = "/output/models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer, model = get_model_from_run(run_cfg, device)

    test_queries = get_query_dataloader(data_cfg.get_query_file(split=data_cfg.test_name), batch_size=16, shuffle=False)
    qrels = get_qrels(data_cfg.get_qrels_file(split=data_cfg.test_name))
    
    if has_embeddings(run_cfg, data_cfg, embs_dir):
        psg_embs, psg_ids = load_embeddings(run_cfg, data_cfg, embs_dir, rel_mode, device)
    else:
        psg_embs, psg_ids = get_embeddings(run_cfg, data_cfg, tokenizer, model, embs_dir, rel_mode, device)

    index = FaissIndex.build(psg_embs)
    
    evaluator = Evaluator(tokenizer, model, rel_mode, device, index=index,
        metrics={"ndcg", "recip_rank"}, psg_ids=psg_ids)
    
    metrics = evaluator.evaluate_retriever(test_queries, qrels, k=10)
    
    logger.info(f"nDCG@{10}: {metrics[f"nDCG@{10}"]}")
    logger.info(f"MRR@{10}: {metrics[f"MRR@{10}"]}")


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    args = OmegaConf.load('config.yml')

    run_cfg = RunConfig(args)
    data_cfg = DatasetConfig(args.eval.dataset_id)

    logger.info(f"Run ID: {args.wandb.run_id}")
    logger.info(f"Dataset id: {args.eval.dataset_id}")
    logger.info(f"Relevance mode: {args.eval.rel_mode}")

    main(run_cfg, data_cfg, rel_mode=args.eval.rel_mode, embs_dir=args.eval.embs_dir)