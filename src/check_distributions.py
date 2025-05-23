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

def metrics(cov: torch.Tensor):
    """
    Compute metrics for the covariance matrix.
    """
    print("Norm", torch.norm(cov, p=2).item())
    print("Mean", torch.mean(cov).item())
    print("Max", torch.max(cov).item())
    print("Min", torch.min(cov).item())

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
        _, model = vbll_model_factory(params.model_name, 1, params.parameterization, params.prior_scale, params.wishart_scale, device)
    else:
        raise ValueError("Only vbll model is supported for evaluation")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print('Weight metrics')
    metrics(model.vbll_layer.W().scale)

    print('Noise metrics')
    metrics(model.vbll_layer.noise().scale)


if __name__ == '__main__':
    args = OmegaConf.load('config.yml')
    data_cfg = DatasetConfig(args.prepare_data.dataset_id)
    api = wandb.Api()
    config = api.run(f"{args.wandb.entity}/{args.wandb.project}/{args.wandb.run_id}").config
    params = SimpleNamespace(**config)
    main(params, data_cfg, args.wandb.run_id)