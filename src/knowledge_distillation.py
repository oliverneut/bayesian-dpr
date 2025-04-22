from omegaconf import OmegaConf
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from models.model_utils import model_factory, vbll_model_factory
from data_loaders import get_dataloader, get_qrels, get_corpus_dataloader, get_query_dataloader
from losses import BinaryPassageRetrievalLoss
from evaluation import Evaluator
from indexing import FaissIndex
from encoding import encode_corpus
from utils.data_utils import get_query_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

def make_lr_scheduler_with_warmup(model, training_data, lr, min_lr, num_epochs, warmup_rate):
    optimizer = Adam(model.parameters(), lr=lr)
    num_training_steps = len(training_data) * num_epochs
    warmup_iters = int(warmup_rate * num_training_steps)
    decay_iters = int((1 - warmup_rate) * num_training_steps)
    decay_factor = min_lr / lr
    warmup = LinearLR(optimizer, start_factor=decay_factor, end_factor=1.0, total_iters=warmup_iters)
    decay = LinearLR(optimizer, start_factor=1.0, end_factor=decay_factor, total_iters=decay_iters)
    scheduler = SequentialLR(optimizer, [warmup, decay], [warmup_iters])
    logger.info("Using linear learning rate scheduling with linear warm-up.")
    logger.info(
        "Total training steps: %d | LR warm-up for %d steps. | LR decay for %d steps.",
        num_training_steps,
        warmup_iters,
        decay_iters,
    )
    return optimizer, scheduler

class KnowledgeDistillationTrainer:
    def __init__(self, train_dl, val_queries, val_corpus, qrels, device, args):
        self.tokenizer = None
        self.student_model = None
        self.teacher_model = None
        self.train_dl = train_dl
        self.val_queries = val_queries
        self.val_corpus = val_corpus
        self.qrels = qrels
        self.device = device
        self.loss_func = BinaryPassageRetrievalLoss()
        self.args = args
    
    def train(self, num_epochs, lr, min_lr, warmup_rate, k=20, alpha=1.0):
        optimizer, scheduler = make_lr_scheduler_with_warmup(
            self.student_model, self.train_dl, lr, min_lr, num_epochs, warmup_rate
        )

        max_ndcg = -1.0

        for epoch in range(1, num_epochs + 1):
            self.student_model.train()
            progress_bar = tqdm(self.train_dl, desc="Train loop")

            for qry, pos_psg, neg_psg in progress_bar:
                qry_enc = self.tokenize_query(qry).to(self.device)
                pos_enc = self.tokenize_passage(pos_psg).to(self.device)
                neg_enc = self.tokenize_passage(neg_psg).to(self.device)

                with torch.no_grad():
                    teacher_qry_emb = self.teacher_model(qry_enc)
                    teacher_pos_emb = self.teacher_model(pos_enc)
                    teacher_neg_emb = self.teacher_model(neg_enc)
                
                optimizer.zero_grad()
                student_qry_emb = self.student_model(qry_enc)
                student_pos_emb = self.student_model(pos_enc)
                student_neg_emb = self.student_model(neg_enc)

                task_loss = self.loss_func(student_qry_emb.predictive.loc, student_pos_emb.predictive.loc, student_neg_emb.predictive.loc)

                qry_loss = student_qry_emb.train_loss_fn(teacher_qry_emb)
                pos_loss = student_pos_emb.train_loss_fn(teacher_pos_emb)
                neg_loss = student_neg_emb.train_loss_fn(teacher_neg_emb)

                kd_loss = qry_loss + pos_loss + neg_loss
                loss = alpha * kd_loss + task_loss
                
                loss.backward()
                optimizer.step()
                # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1)
                scheduler.step()
                progress_bar.set_postfix({"Loss": loss.item()})

            ndcg, mrr = self.compute_validation_metrics(k)
            logger.info(f"Epoch {epoch}/{args.num_epochs} ")
            logger.info(f"Validation metrics: nDCG@{k}={ndcg:.4f} | MRR@{k}={mrr:.4f}")

            if ndcg > max_ndcg:
                model_path = os.path.join(args.output_dir, f"{args.ckpt_filename}.pt")
                torch.save(self.student_model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
                max_ndcg = ndcg

    def tokenize_query(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True, max_length=args.max_qry_len, return_tensors="pt")
    
    def tokenize_passage(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True, max_length=args.max_psg_len, return_tensors="pt")

    def set_student_model(self, args):
        model_name = args.student_model_name
        reg_weight = 1.0 / len(self.train_dl)
        prior_scale = args.prior_scale
        wishart_scale = args.wishart_scale
        paremeterization = args.paremeterization
        self.tokenizer, self.student_model = vbll_model_factory(model_name, reg_weight, paremeterization, prior_scale, wishart_scale, self.device)

    def set_teacher_model(self, args):
        model_name = args.teacher_model_name
        _, self.teacher_model = model_factory(model_name, self.device)
        self.teacher_model.eval()

    def compute_validation_metrics(self, k=20):
        self.student_model.eval()
        psg_embs, psg_ids = encode_corpus(self.val_corpus, self.tokenizer, self.student_model, self.device, method="vbll")
        index = FaissIndex.build(psg_embs)
        evaluator = Evaluator(
            self.tokenizer,
            self.student_model,
            "vbll",
            self.device,
            index=index,
            metrics={"ndcg", "recip_rank"},
            psg_ids=psg_ids
        )
        
        metrics = evaluator.evaluate_retriever(self.val_queries, self.qrels, k=k)
        ndcg = metrics[f"nDCG@{k}"]
        mrr = metrics[f"MRR@{k}"]
        
        return ndcg, mrr

def main(args):
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
        
    # Load data
    logger.info("Loading data...")
    train_dataloader = get_dataloader(get_query_file(split="train"), batch_size=args.batch_size, shuffle=True)
    val_queries = get_query_dataloader(get_query_file(split="val"),  batch_size=1, shuffle=False)
    val_corpus = get_corpus_dataloader("data/prepared/corpus-val.jsonl",  batch_size=args.batch_size, shuffle=False)
    qrels = get_qrels()

    kd_trainer = KnowledgeDistillationTrainer(train_dataloader, val_queries, val_corpus, qrels, device, args)
    kd_trainer.set_teacher_model(args)
    kd_trainer.set_student_model(args)

    kd_trainer.train(args.num_epochs, args.lr, args.min_lr, args.warmup_rate, k=args.k, alpha=args.alpha)

    logger.info(f"Training completed after {args.num_epochs} epochs!")

if __name__ == '__main__':
    args = OmegaConf.load('src/utils/config.yml').knowledge_distillation
    main(args)