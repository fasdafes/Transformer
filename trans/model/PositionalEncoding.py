import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Positional Encoding lets the model know where elements are supposed to be
by giving them sequences  
PE(2i) = sin(POS/10000^(2i/embed_dim))
PE(2i+1) = cos(POS/10000^(2i/embed_dim))

pos：序列中的位置，从 0 开始递增（表示是第几个 token）
i：编码维度索引（embedding 的第几个维度）
2i、2i+1：表示分别用 sin 和 cos 交错填充向量
10000^(2𝑖/embed_dim)
 ：调节不同维度的频率，低维频率高，高维频率低
"""
class PositionalEncoding(nn.Module):
    def __init__(self,embed_dim,max_len = 5000):
        super().__init__()

        pe = torch.zeros(max_len,embed_dim)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        divide = torch.exp(torch.arange(0,embed_dim,2).float()*(-math.log(10000.0)/embed_dim))

        pe[:,0::2] = torch.sin(position *divide)
        pe[:,1::2] = torch.cos(position *divide)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        seq_len = x.size(1)
        return x+self.pe[:,:seq_len]