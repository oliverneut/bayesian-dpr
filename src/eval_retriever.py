from omegaconf import OmegaConf
import logging
import os
import torch
from models.model_utils import vbll_model_factory
from data_loaders import get_qrels, get_corpus_dataloader, get_query_dataloader
from utils.data_utils import get_query_file
from encoding import encode_corpus
from evaluation import Evaluator
from indexing import FaissIndex

logger = logging.getLogger(__name__)

def main(kd_args, eval_args):
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    
    # Create output directory
    os.makedirs(eval_args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer, model = vbll_model_factory(kd_args.student_model_name, 1, kd_args.paremeterization, kd_args.prior_scale, kd_args.wishart_scale, device)

    # model_path = os.path.join(kd_args.output_dir, kd_args.ckpt_filename)
    # model.load_state_dict(torch.load(model_path))
    # model.eval()

    test_queries = get_query_dataloader(get_query_file(split="dev"),  batch_size=eval_args.batch_size, shuffle=False)
    corpus = get_corpus_dataloader("data/msmarco/corpus.jsonl",  batch_size=eval_args.batch_size, shuffle=False)
    qrels = get_qrels(split="dev")

    psg_embs, psg_ids = encode_corpus(corpus, tokenizer, model, device, method="vbll")
    index = FaissIndex.build(psg_embs)
    evaluator = Evaluator(tokenizer, model, "vbll", device, index=index,
        metrics={"ndcg", "recip_rank"}, psg_ids=psg_ids)
    
    metrics = evaluator.evaluate_retriever(test_queries, qrels, k=eval_args.k)
    ndcg = metrics[f"nDCG@{eval_args.k}"]
    mrr = metrics[f"MRR@{eval_args.k}"]

    logger.info(f"nDCG@{eval_args.k}: {ndcg}")
    logger.info(f"MRR@{eval_args.k}: {mrr}")


if __name__ == '__main__':
    kd_args = OmegaConf.load('src/utils/config.yml').knowledge_distillation
    eval_args = OmegaConf.load('src/utils/config.yml').eval_retriever
    main(kd_args, eval_args)