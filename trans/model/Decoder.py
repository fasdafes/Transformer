import torch
import torch.nn as nn
import torch.nn.functional as F
from .MultiHeadSelfAttention import MultiHeadSelfAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self,embed_dim,nums_head,ff_hidden_dim = None,dropout = 0.1):
        super().__init__()
        #masked self attetion
        self.self_attn1 = MultiHeadSelfAttention(embed_dim,nums_head,masked=True)
        #cross self attetion
        self.self_attn2 = MultiHeadSelfAttention(embed_dim,nums_head,masked=False)
        #layerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        ff_hidden_dim = ff_hidden_dim if ff_hidden_dim is not None else embed_dim*4
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim,ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim,embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self,x,encoder_output):
        #masked self_attention
        self_attn_out = self.self_attn1(x)
        x = self.norm1(x+self.dropout(self_attn_out))# 残差连接（Residual Connection）是 element-wise 加法，不是拼接

        #cross self_attention
        self_attn_out2 = self.self_attn2(x,encoder_output)
        x = self.norm2(x+self.dropout(self_attn_out2))

        #feed_forward
        ffn_out = self.feed_forward(x)
        x = self.norm3(x+self.dropout(ffn_out))

        return x