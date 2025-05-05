import torch.nn as nn
import torch
from .PositionalEncoding import PositionalEncoding
import traceback

class EmbeddingWithPE(nn.Module):
    def __init__(self,vocab_size,embed_dim,max_len=5000):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size,embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim,max_len)

    def forward(self,token_ids):
        if token_ids.dtype != torch.long:
            print("🟥 FLOAT 类型嵌入触发！自动转换为 long！")
            token_ids = token_ids.long()

        x = self.token_embed(token_ids)
        x = self.pos_embed(x)
        return x