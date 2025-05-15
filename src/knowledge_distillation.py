from omegaconf import OmegaConf
import logging
import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from utils.model_utils import model_factory, vbll_model_factory, get_model_save_path
from data_loaders import get_dataloader, get_qrels, get_corpus_dataloader, get_query_dataloader
from losses import BinaryPassageRetrievalLoss
from evaluation import Evaluator
from indexing import FaissIndex
from encoding import encode_corpus
from utils.data_utils import get_query_file
from tqdm import tqdm
import wandb
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

class DPRTrainer:
    def __init__(self, train_dl, val_queries, val_corpus, qrels, run, device, save_path, args):
        self.tokenizer = None
        self.model = None
        self.train_dl = train_dl
        self.val_queries = val_queries
        self.val_corpus = val_corpus
        self.qrels = qrels
        self.run = run
        self.device = device
        self.save_path = save_path
        self.method = "dpr"
        self.loss_func = BinaryPassageRetrievalLoss()
        self.args = args

    def train(self, num_epochs, lr, min_lr, warmup_rate, k=20):
        optimizer, scheduler = make_lr_scheduler_with_warmup(
            self.model, self.train_dl, lr, min_lr, num_epochs, warmup_rate
        )

        max_ndcg = -1.0

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            progress_bar = tqdm(self.train_dl, desc="Train loop")

            for qry, pos_psg, neg_psg in progress_bar:
                qry_enc = self.tokenize_query(qry).to(self.device)
                pos_enc = self.tokenize_passage(pos_psg).to(self.device)
                neg_enc = self.tokenize_passage(neg_psg).to(self.device)

                optimizer.zero_grad()
                qry_emb = self.model(qry_enc)
                pos_emb = self.model(pos_enc)
                neg_emb = self.model(neg_enc)

                task_loss = self.loss_func(qry_emb, pos_emb, neg_emb)
                loss = task_loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                progress_bar.set_postfix({"Loss": loss.item()})
                self.run.log({"kd_loss": 0, "task_loss": task_loss.item(), "loss": loss.item()})

            ndcg, mrr = self.compute_validation_metrics(k)
            logger.info(f"Epoch {epoch}/{args.num_epochs} ")
            logger.info(f"Validation metrics: nDCG@{k}={ndcg:.4f} | MRR@{k}={mrr:.4f}")

            if ndcg > max_ndcg:
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f"Model saved to {self.save_path}")
                max_ndcg = ndcg

    def tokenize_query(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True, max_length=self.args.max_qry_len, return_tensors="pt")
    
    def tokenize_passage(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True, max_length=self.args.max_psg_len, return_tensors="pt")
    
    def set_model(self):
        self.tokenizer, self.model = model_factory(self.args.model_name, self.device)

    def compute_validation_metrics(self, k=20):
        self.model.eval()
        psg_embs, psg_ids = encode_corpus(self.val_corpus, self.tokenizer, self.model, self.device, method=self.method)
        index = FaissIndex.build(psg_embs)
        evaluator = Evaluator(
            self.tokenizer,
            self.model,
            self.method,
            self.device,
            index=index,
            metrics={"ndcg", "recip_rank"},
            psg_ids=psg_ids
        )
        
        metrics = evaluator.evaluate_retriever(self.val_queries, self.qrels, k=k)
        ndcg = metrics[f"nDCG@{k}"]
        mrr = metrics[f"MRR@{k}"]
        
        return ndcg, mrr


class KnowledgeDistillationTrainer(DPRTrainer):
    def __init__(self, train_dl, val_queries, val_corpus, qrels, run, device, save_path, args):
        super().__init__(train_dl, val_queries, val_corpus, qrels, run, device, save_path, args)
        self.teacher_model = None
        self.method = "vbll"
    
    def train(self, num_epochs, lr, min_lr, warmup_rate, k=20, alpha=1.0):
        optimizer, scheduler = make_lr_scheduler_with_warmup(
            self.model, self.train_dl, lr, min_lr, num_epochs, warmup_rate
        )
        max_mrr = -1.0
        patience = 3

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            progress_bar = tqdm(self.train_dl, desc="Train loop")

            for qry, pos_psg, neg_psg in progress_bar:
                qry_enc = self.tokenize_query(qry).to(self.device)
                pos_enc = self.tokenize_passage(pos_psg).to(self.device)
                neg_enc = self.tokenize_passage(neg_psg).to(self.device)
                
                optimizer.zero_grad()
                qry_emb = self.model(qry_enc)
                pos_emb = self.model(pos_enc)
                neg_emb = self.model(neg_enc)

                task_loss = self.loss_func(qry_emb.predictive.loc, pos_emb.predictive.loc, neg_emb.predictive.loc)
                
                if alpha > 0:
                    with torch.no_grad():
                            teacher_qry_emb = self.teacher_model(qry_enc)
                            teacher_pos_emb = self.teacher_model(pos_enc)
                            teacher_neg_emb = self.teacher_model(neg_enc)
                    qry_loss = qry_emb.train_loss_fn(teacher_qry_emb)
                    pos_loss = pos_emb.train_loss_fn(teacher_pos_emb)
                    neg_loss = neg_emb.train_loss_fn(teacher_neg_emb)
                    kd_loss = qry_loss + pos_loss + neg_loss
                else:
                    kd_loss = torch.tensor(0.0)

                loss = alpha * kd_loss + task_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                progress_bar.set_postfix({"Loss": loss.item()})
                self.run.log({"kd_loss": kd_loss.item(), "task_loss": task_loss.item(), "loss": loss.item()})
                break
            
            ndcg, mrr = 0, 0
            # ndcg, mrr = self.compute_validation_metrics(k)
            logger.info(f"Epoch {epoch}/{self.args.num_epochs} ")
            logger.info(f"Validation metrics: nDCG@{k}={ndcg:.4f} | MRR@{k}={mrr:.4f}")
            self.run.log({f"nDCG@{k}": ndcg, f"MRR@{k}": mrr})

            if mrr > max_mrr:
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f"Model saved to {self.save_path}")
                max_mrr = mrr
                patience = 3
            else:
                patience -= 1

            if patience <= 0:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    def set_model(self):
        model_name = self.args.model_name
        reg_weight = 1.0 / len(self.train_dl.dataset)
        prior_scale = self.args.prior_scale
        wishart_scale = self.args.wishart_scale
        parameterization = self.args.parameterization
        self.tokenizer, self.model = vbll_model_factory(model_name, reg_weight, parameterization, prior_scale, wishart_scale, self.device)

    def set_teacher_model(self):
        _, self.teacher_model = model_factory(self.args.teacher_model_name, self.device)
        self.teacher_model.eval()

def main(args, run):
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    
    # Create output directory
    save_dir = f"{args.output_dir}/{run.id}"
    save_path = f"{save_dir}/model.pt"
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_dataloader = get_dataloader(get_query_file(split="train"), batch_size=args.batch_size, shuffle=True)
    val_queries = get_query_dataloader(get_query_file(split="val"),  batch_size=args.batch_size, shuffle=False)
    val_corpus = get_corpus_dataloader("data/prepared/corpus-val.jsonl",  batch_size=args.batch_size, shuffle=False)
    qrels = get_qrels()

    if args.knowledge_distillation:
        trainer = KnowledgeDistillationTrainer(train_dataloader, val_queries, val_corpus, qrels, run, device, save_path, args)
        trainer.set_teacher_model()
        trainer.set_model()
        trainer.train(args.num_epochs, args.lr, args.min_lr, args.warmup_rate, k=args.k, alpha=args.alpha)
    else:
        trainer = DPRTrainer(train_dataloader, val_queries, val_corpus, qrels, run, device, save_path, args)
        trainer.set_model()
        trainer.train(args.num_epochs, args.lr, args.min_lr, args.warmup_rate, k=args.k)

    trainer.run.finish()

    logger.info(f"Training completed after {args.num_epochs} epochs!")

if __name__ == '__main__':
    args = OmegaConf.load('src/utils/config.yml')
    run = wandb.init(entity=args.wandb.entity, project=args.wandb.project, config=OmegaConf.to_container(args.train))
    main(args.train, run)