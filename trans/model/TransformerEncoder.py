import torch.nn as nn
from .Encoder import TransformerEncoderLayer
from .EmbeddingWithPE import EmbeddingWithPE
class TransformerEncoder(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_layers,num_heads,ff_hidden_dim,dropout=0.1,max_len=5000):
        super().__init__()
        self.embed_pe = EmbeddingWithPE(vocab_size,embed_dim,max_len)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim,num_heads,ff_hidden_dim,dropout = dropout)
            for _ in range(num_layers)
        ])
    def forward(self,src):
        x = self.embed_pe(src)
        for layer in self.layers:
            x = layer(x)
        return x  #<------this is Key and Value for the cross attention
