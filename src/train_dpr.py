import argparse
import logging

import torch

from utils.model_utils import model_factory
from data_loaders import get_dataloader, get_qrels, get_corpus_dataloader, get_query_dataloader
from utils.data_utils import get_query_file
from trainer import DPRTrainer

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = model_factory(args.model_name, device)

    if args.encoder_ckpt is not None:
        sd = torch.load(args.encoder_ckpt)
        model.load_state_dict(sd)
    
    model.train()

    train_query_dl = get_dataloader(get_query_file(split="train"), batch_size=args.batch_size, shuffle=True)
    val_query_dl = get_query_dataloader(get_query_file(split="val"), batch_size=1, shuffle=False)
    val_corpus_dl = get_corpus_dataloader("data/prepared/corpus-val.jsonl", batch_size=1, shuffle=False)
    qrels = get_qrels()

    ckpt_file_name = "models/bert-tiny.pt"

    trainer = DPRTrainer(tokenizer, model, train_query_dl, val_query_dl, val_corpus_dl, qrels, device)
    trainer.train(
        num_epochs=args.num_epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_rate=args.warmup_rate,
        ckpt_file_name=ckpt_file_name,
        num_samples=args.num_samples,
        max_qry_len=args.max_qry_len,
        max_psg_len=args.max_psg_len,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", choices=["msmarco"])
    parser.add_argument("--model_name", default="bert-tiny")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--encoder_ckpt", default=None)  # If provided, training is resumed from checkpoint.
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--min_lr", type=float, default=5e-8)
    parser.add_argument("--warmup_rate", type=float, default=0.1)
    parser.add_argument("--max_qry_len", type=int, default=32)
    parser.add_argument("--max_psg_len", type=int, default=256)
    parser.add_argument("--output_dir", default="output/trained_encoders")
    args = parser.parse_args()
    main(args)