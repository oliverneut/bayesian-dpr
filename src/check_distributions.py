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


def log_metric(name: str, metrics: torch.Tensor):
    if metrics.shape[0] == 1:
        print(f"{name}: N({metrics.item()}, 0.0)")
        return
        
    mean = torch.mean(metrics)
    std = torch.std(metrics)

    print(f"{name}: N({mean.item()}, {std.item()})")
    

def metrics(cov: torch.Tensor):
    """
    Compute metrics for the covariance matrix.
    """
    log_metric("Norm", torch.norm(cov, p=2, dim=1))
    log_metric("Mean", torch.mean(cov, dim=1))
    if cov.shape[0] > 1:
        log_metric("Std", torch.std(cov, dim=0))
    log_metric("Max", torch.max(cov, dim=1).values)
    log_metric("Min", torch.min(cov, dim=1).values)



def main(params: SimpleNamespace, data_cfg: DatasetConfig, run_id: str):
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset id: {data_cfg.dataset_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    save_dir = f"output/models/{run_id}"
    model_path = f"{save_dir}/model.pt"
    
    if params.knowledge_distillation:
        tokenizer, model = vbll_model_factory(params.model_name, 1, params.parameterization, params.prior_scale, params.wishart_scale, device)
    else:
        raise ValueError("Only vbll model is supported for evaluation")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print('Noise metrics')
    metrics(model.vbll_layer.noise().scale.unsqueeze(dim=0))

    test_queries = get_query_dataloader(data_cfg.get_query_file(split=data_cfg.test_name), batch_size=params.batch_size, shuffle=False)

    for _, query in test_queries:
        qry_enc = tokenizer(query, padding="max_length", truncation=True, max_length=params.max_qry_len, return_tensors="pt").to(device)

        qry_emb = model(qry_enc).predictive
        qry_emb_cov = qry_emb.scale

        print('Covariance metrics')
        metrics(qry_emb_cov)
        break


if __name__ == '__main__':
    args = OmegaConf.load('src/utils/config.yml')
    data_cfg = DatasetConfig(args.prepare_data.dataset_id)
    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    params = SimpleNamespace(**config)
    main(params, data_cfg, args.wandb.run_id)