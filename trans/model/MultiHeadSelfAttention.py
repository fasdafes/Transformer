import torch
import torch.nn as nn
import torch.nn.functional as F
from trans.utils.utils import generate_causal_mask


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embed_dim,num_head,masked = False):
        super().__init__()
        assert  embed_dim % num_head == 0 #确保embed结果维度可以被头数量整除

        self.embed_dim = embed_dim
        self.num_heads = num_head
        self.head_dim = embed_dim//num_head
        self.masked = masked

        self.q_project = nn.Linear(embed_dim,embed_dim)  #q_project是一个让输入从X到Q结果的线性变换，包括weights和bios
        self.k_project = nn.Linear(embed_dim,embed_dim)  #同上
        self.v_project = nn.Linear(embed_dim,embed_dim)  #同上

        self.out_project = nn.Linear(embed_dim,embed_dim) #模块输出前的W_O 让结果重新映射到embed空间
        #以上四个线性变换都可以训练

    def forward(self,x,kv_input = None):
        #batch_size,seq_len,embed_dim = x.size()
        #x = torch.randn(32, 10, 512)
        #说明你现在有：
        #一个 batch 中 32 个句子
        #每个句子有 10 个 token
        # 每个 token 被 embedding 成了 512 维向量

        ###添加Cross-attention支持，只需要再加一个参数，kv_input
        Q_input = x
        K_input = V_input = x if kv_input is None else kv_input
        #如果kv_input不为空，则证明是cross-attention，需要分开处理QKV
        batch_size = Q_input.size(0)
        q_len = Q_input.size(1)
        kv_len = K_input.size(1)


        Q = self.q_project(Q_input) #注意 PyTorch 里 nn.Linear 会自动帮你做 .matmul(x, W^T) + bias
        K = self.k_project(K_input)
        V = self.v_project(V_input)


        #拆分成多头
        Q = Q.view(batch_size,q_len,self.num_heads,self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size,kv_len,self.num_heads,self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size,kv_len,self.num_heads,self.head_dim).permute(0,2,1,3)
        # shape: (batch_size, num_heads, seq_len, head_dim)
        # 表示：
        # - batch_size: 一次送入模型的句子数量（即有多少句话）
        # - num_heads: 每个 token 的 embedding 被分成多少个 attention head（注意力头）
        # - seq_len: 每句话中 token 的个数
        # - head_dim: 每个 attention 头处理的子空间维度（即每头“看到”的维度）

        score = torch.matmul(Q,K.transpose(-2,-1))/self.head_dim**(0.5)# (batch, heads, seq, seq)
        if self.masked==True:
            assert kv_input is None #判断是否是cross-attention，如果是的话不允许进行掩码

            #如果是masked的
            #引入utils的下三角形矩阵生辰函数
            mask = generate_causal_mask(q_len,device=x.device)
            score = score.masked_fill(~mask.unsqueeze(0).unsqueeze(0),float('-inf'))
            #~mask布尔取反 True——》False  False——》True
            #unsqueeze(0)，在前面再加一个维度
            #masked_fill就是填充内容，True的地方填充-inf   -inf = - infinity
        #计算权重
        attn_weights = F.softmax(score,dim = -1)
        #计算加权
        context = torch.matmul(attn_weights,V)
        #拼接多个头
        context = context.permute(0,2,1,3).reshape(batch_size,q_len,self.embed_dim)
        #最后的线性变换
        output = self.out_project(context)

        return output

#test----------------------------------------------------------
x = torch.randn(2, 5, 512)
mha = MultiHeadSelfAttention(embed_dim=512, num_head=8, masked=False)
out = mha(x)
print(out.shape)  # → (2, 5, 512)
print("======================================================")
mha_masked = MultiHeadSelfAttention(embed_dim=512, num_head=8, masked=True)
out_masked = mha_masked(x)
print(out_masked.shape)  # → (2, 5, 512)
print("======================================================")
decoder_input = torch.randn(2, 3, 512)
encoder_output = torch.randn(2, 7, 512)
mha_cross = MultiHeadSelfAttention(embed_dim=512, num_head=8, masked=False)
out_cross = mha_cross(decoder_input, kv_input=encoder_output)
print(out_cross.shape)  # → (2, 3, 512)
