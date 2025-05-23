import argparse
from omegaconf import OmegaConf
from utils.model_utils import get_model_save_path, vbll_model_factory, model_factory
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    model_path = get_model_save_path(args.output_dir, args.ckpt_filename, args.alpha, args.prior_scale, args.wishart_scale)

    if args.knowledge_distillation:
        _, model = vbll_model_factory(args.model_name, 1, args.parameterization, args.prior_scale, args.wishart_scale, device)
    else:
        _, model = model_factory(args.model_name, device)
    
    model.load_state_dict(torch.load(model_path))
    model.push_to_hub(args.repo_name)

if __name__ == "__main__":
    args = OmegaConf.load('config.yml').upload
    main(args)

