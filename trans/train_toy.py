
import torch
import torch.nn as nn
import torch.optim as optim
from trans.toy_dataset import build_toy_dataloader
from trans.model.Transformer import Transformer  # 修改为你的实际路径
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建 dataloader 和词表
dataloader, src_vocab, tgt_vocab = build_toy_dataloader(batch_size=2)

# 初始化模型
model = Transformer(
    vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    embed_dim=256,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    ff_hidden_dim=512,
    dropout=0.1,
    max_len=100
).to(device)

# 损失和优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 多轮训练
for epoch in range(30):
    model.train()
    total_loss = 0
    for batch in dataloader:
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

# 保存模型
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/transformer_toy.pth")
print("✅ 模型已保存到 checkpoints/transformer_toy.pth")
