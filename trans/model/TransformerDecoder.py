import torch.nn as nn
from .Decoder import TransformerDecoderLayer
from .EmbeddingWithPE import EmbeddingWithPE
class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_layers,num_heads,ff_hidden_dim,dropout = 0.1,max_len = 5000):
        super().__init__()
        self.embed = EmbeddingWithPE(vocab_size,embed_dim,max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim,num_heads,ff_hidden_dim,dropout)
            for _ in range(num_layers)
        ])

    def forward(self,tgt,encoder_output):
        """tgt is the changed Ground Truth"""
        x = self.embed(tgt)

        for layer in self.layers:
            x = layer(x,encoder_output)
        return x