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

def main(args):
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    k = args.num_samples # ndcg@k and mrr@k
    a = args.a # weight of KD-loss
        
    # Load data
    logger.info("Loading data...")
    train_dataloader = get_dataloader(get_query_file(split="train"), batch_size=args.batch_size, shuffle=True)
    val_queries = get_query_dataloader(get_query_file(split="val"),  batch_size=1, shuffle=False)
    val_corpus = get_corpus_dataloader("data/prepared/corpus-val.jsonl",  batch_size=args.batch_size, shuffle=False)
    qrels = get_qrels()

    # Load teacher model
    logger.info(f"Loading teacher model: sentence-transformers/msmarco-bert-base-dot-v5")
    teacher_tokenizer, teacher_model = model_factory("bert-base-msmarco", device)
    teacher_model.eval()  # Set teacher to evaluation mode
    
    # Load student model
    logger.info(f"Loading student model: {args.model_name}")
    reg_weight = 1.0 / len(train_dataloader)
    prior_scale = 1.0
    wishart_scale = 0.1
    paremeterization = 'diagonal'
    student_tokenizer, student_model = vbll_model_factory(args.model_name, reg_weight, paremeterization, prior_scale, wishart_scale, device)
    
    # Print model sizes
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"Teacher model parameters: {teacher_params:,}")
    logger.info(f"Student model parameters: {student_params:,}")
    logger.info(f"Compression ratio: {student_params/teacher_params:.2f}")
    
    # Set up loss function
    task_loss_func = BinaryPassageRetrievalLoss()
    
    # Set up optimizer and scheduler
    optimizer, scheduler = make_lr_scheduler_with_warmup(
        student_model, train_dataloader, args.lr, args.min_lr, args.num_epochs, args.warmup_rate
    )
    
    # Training loop
    logger.info("Starting training...")
    best_ndcg = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        student_model.train()
        epoch_loss = 0.0

        batch_idx = 0
        progress_bar = tqdm(train_dataloader, desc="Train loop")
        for qry, pos_psg, neg_psg in progress_bar:
            # Process query batch
            qry_enc = student_tokenizer(
                qry, padding="max_length", truncation=True, 
                max_length=args.max_qry_len, return_tensors="pt"
            ).to(device)
            
            # Process passage batches
            pos_enc = student_tokenizer(
                pos_psg, padding="max_length", truncation=True,
                max_length=args.max_psg_len, return_tensors="pt"
            ).to(device)

            neg_enc = student_tokenizer(
                neg_psg, padding="max_length", truncation=True,
                max_length=args.max_psg_len, return_tensors="pt"
            ).to(device)
            
            # Forward pass with teacher model (no gradient)
            with torch.no_grad():
                teacher_qry_emb = teacher_model(qry_enc)
                teacher_pos_emb = teacher_model(pos_enc)
                teacher_neg_emb = teacher_model(neg_enc)
            
            # Forward pass with student model
            optimizer.zero_grad()
            student_qry_emb = student_model(qry_enc)
            student_pos_emb = student_model(pos_enc)
            student_neg_emb = student_model(neg_enc)
            
            # Compute loss
            qry_loss = student_qry_emb.train_loss_fn(teacher_qry_emb)
            pos_loss = student_pos_emb.train_loss_fn(teacher_pos_emb)
            neg_loss = student_neg_emb.train_loss_fn(teacher_neg_emb)

            # task_loss = task_loss_func(student_qry_emb.predictive.loc, student_pos_emb.predictive.loc, student_neg_emb.predictive.loc)
            task_loss = 0
            kd_loss = qry_loss + pos_loss + neg_loss
            # loss = a * kd_loss +  task_loss
            loss = a * kd_loss
                        
            loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1)
            scheduler.step()
            
            epoch_loss += loss.item()
            # progress_bar.set_postfix({"Loss": loss.item(), "KD loss": kd_loss.item(), "Task loss": task_loss.item()})
            progress_bar.set_postfix({"Loss": loss.item()})

            batch_idx += 1
            if batch_idx / len(train_dataloader) >= 0.1:
                break
        
        # Evaluate on validation set
        student_model.eval()
        psg_embs, psg_ids = encode_corpus(val_corpus, student_tokenizer, student_model, device, method="vbll")
        index = FaissIndex.build(psg_embs)
        evaluator = Evaluator(
            student_tokenizer,
            student_model,
            "vbll",
            device,
            index=index,
            metrics={"ndcg", "recip_rank"},
            psg_ids=psg_ids
        )
        
        metrics = evaluator.evaluate_retriever(val_queries, qrels, k=k)
        ndcg = metrics[f"nDCG@{k}"]
        mrr = metrics[f"MRR@{k}"]
        
        logger.info(f"Epoch {epoch}/{args.num_epochs} - Loss: {epoch_loss/len(train_dataloader):.4f} - nDCG@{k}: {ndcg:.4f} - MRR@{k}: {mrr:.4f}")
        
        # Save best model
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            model_path = os.path.join(args.output_dir, f"{args.model_name}-distilled.pt")
            torch.save(student_model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
    
    logger.info("Training completed!")

if __name__ == '__main__':
    args = OmegaConf.load('src/utils/config.yml').knowledge_distillation
    main(args)