import torch
from tqdm import tqdm
from vbll.layers.regression import VBLLReturn

def encode_corpus(corpus, tokenizer, encoder, device, max_psg_len=256):
    psg_embs = []
    psg_ids = []
    
    with torch.no_grad():
        for psg_id, psg in tqdm(corpus, desc="Encoding corpus"):
            psg_enc = tokenizer(psg, padding="max_length", truncation=True, max_length=max_psg_len, return_tensors="pt").to(device)

            psg_emb = encoder(psg_enc)

            if isinstance(psg_emb, VBLLReturn):
                psg_emb = psg_emb.predictive
                psg_emb = torch.stack([psg_emb.mean, psg_emb.variance], dim=1)

            psg_embs.append(psg_emb.detach().cpu())
            psg_ids += list(psg_id)
            break

        psg_embs = torch.cat(psg_embs, dim=0)
    return psg_embs, psg_ids