import torch
import torch.nn as nn
from .trf import TransformerBlock, LayerNorm

class GPT_MODEL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, input_ids):
        batch_size, input_len = input_ids.shape
        token_embeds = self.token_emb(input_ids)
        pos_embeds = self.pos_emb(torch.arange(input_len, device=input_ids.device))
        x = token_embeds + pos_embeds

        x = self.drop_emb(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
