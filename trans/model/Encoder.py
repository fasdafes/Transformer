import torch
import torch.nn as nn
import torch.nn.functional as F
from .MultiHeadSelfAttention import MultiHeadSelfAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,ff_hidden_dim=None,dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim,num_heads,masked=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        #LayerNorm Normalization
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        ff_hidden_dim = ff_hidden_dim or embed_dim * 4
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim,ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim,embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        # Self-attention + 残差 + layerNorm
        attn_out = self.self_attn(x)
        x = self.norm1(x+self.dropout(attn_out))

        # FeedForward(fc + GeLu + fc +dropout) + 残差 +LayerNorm
        ff_out = self.feed_forward(x)
        x = self.norm2(x+self.dropout(ff_out))

        return x