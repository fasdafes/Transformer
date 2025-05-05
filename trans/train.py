import torch
import torch.nn as nn
import torch.optim as optim
from trans.model.Transformer import Transformer

# ----------------------------
# 超参数
# ----------------------------
vocab_size = 1000       # 假设英法共用词表
embed_dim = 512
num_heads = 8
num_encoder_layers = 2
num_decoder_layers = 2
ff_hidden_dim = 2048
dropout = 0.1
max_len = 100
batch_size = 2
src_len = 10
tgt_len = 9  # decoder 输入长度（输出长度为9，真实target为右移版本）

# ----------------------------
# 初始化模型、损失函数、优化器
# ----------------------------
model = Transformer(
    vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    ff_hidden_dim=ff_hidden_dim,
    dropout=dropout,
    max_len=max_len
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0为 <pad> token
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# 生成 dummy 训练数据（英-法对照）
# ----------------------------
# 输入：英语句子
src = torch.randint(3, vocab_size, (batch_size, src_len)).to(device)
# 输入给 decoder 的序列（法语句子前面的部分）
tgt_input = torch.randint(3, vocab_size, (batch_size, tgt_len)).to(device)
# 目标输出（tgt_input 的右边部分，用来计算 loss）
tgt_output = torch.randint(3, vocab_size, (batch_size, tgt_len)).to(device)

# 可选：第一个 token 设为 <bos>，最后一个为 <eos>，pad 设为 0
# src[:, -1] = 0  # 末尾 padding 模拟

# ----------------------------
# 前向传播
# ----------------------------
num_epochs = 10  # 你可以改成更大的，比如 100、500

for epoch in range(num_epochs):
    model.train()

    # 生成新的 dummy 数据（可选）
    src = torch.randint(3, vocab_size, (batch_size, src_len)).to(device)
    tgt_input = torch.randint(3, vocab_size, (batch_size, tgt_len)).to(device)
    tgt_output = torch.randint(3, vocab_size, (batch_size, tgt_len)).to(device)

    logits = model(src, tgt_input)
    loss = criterion(logits.view(-1, vocab_size), tgt_output.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

