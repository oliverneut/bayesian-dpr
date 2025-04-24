from omegaconf import OmegaConf
import logging
import os
import torch
from utils.model_utils import vbll_model_factory, model_factory, get_model_save_path
from data_loaders import get_qrels, get_corpus_dataloader, get_query_dataloader
from utils.data_utils import get_query_file
from encoding import encode_corpus
from evaluation import Evaluator
from indexing import FaissIndex

logger = logging.getLogger(__name__)

def main(train, eval):
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    
    model_path = get_model_save_path(train.output_dir, train.ckpt_filename, train.alpha, train.prior_scale, train.wishart_scale)

    # Log evaluation parameters
    logger.info("Starting evaluation with parameters:")
    logger.info(f"  Model: {train.model_name}")
    logger.info(f"  Batch size: {train.batch_size}")
    logger.info(f"  k: {train.k}")
    logger.info(f"  alpha: {train.alpha}")
    logger.info(f"  prior scale: {train.prior_scale}")
    logger.info(f"  wishart scale: {train.wishart_scale}")
    logger.info(f"  lr: {train.lr}")
    logger.info(f"  min lr: {train.min_lr}")
    logger.info(f"  ckpt filepath: {model_path}")
    
    # Create output directory
    os.makedirs(eval.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if train.knowledge_distillation:
        tokenizer, model = vbll_model_factory(train.model_name, 1, train.paremeterization, train.prior_scale, train.wishart_scale, device)
        method = "vbll"
    else:
        tokenizer, model = model_factory(train.model_name, device)
        method = "dpr"

    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_queries = get_query_dataloader(get_query_file(split="dev"),  batch_size=eval.batch_size, shuffle=False)
    corpus = get_corpus_dataloader("data/msmarco/corpus.jsonl",  batch_size=eval.batch_size, shuffle=False)
    qrels = get_qrels(split="dev")

    psg_embs, psg_ids = encode_corpus(corpus, tokenizer, model, device, method=method)
    index = FaissIndex.build(psg_embs)
    evaluator = Evaluator(tokenizer, model, method, device, index=index,
        metrics={"ndcg", "recip_rank"}, psg_ids=psg_ids)
    
    metrics = evaluator.evaluate_retriever(test_queries, qrels, k=train.k)
    ndcg = metrics[f"nDCG@{train.k}"]
    mrr = metrics[f"MRR@{train.k}"]

    logger.info(f"nDCG@{train.k}: {ndcg}")
    logger.info(f"MRR@{train.k}: {mrr}")


if __name__ == '__main__':
    args = OmegaConf.load('src/utils/config.yml')
    main(args.train, args.eval)