
import torch
from trans.toy_dataset import build_vocab, tokenize, encode
from trans.model.Transformer import Transformer  # 修改为你的路径
import torch.nn.functional as F

# 同 toy_dataset 中的训练数据
eng_sentences = [
    "i am a student",
    "he is a teacher",
    "she is a nurse",
    "you are a boy",
    "they are girls",
    "hello",
    "good morning",
    "thank you",
    "how are you",
    "i am fine"
]

fra_sentences = [
    "je suis un étudiant",
    "il est un professeur",
    "elle est une infirmière",
    "tu es un garçon",
    "elles sont des filles",
    "bonjour",
    "bonjour",
    "merci",
    "comment ça va",
    "je vais bien"
]

# 构建词表
src_vocab = build_vocab(eng_sentences)
tgt_vocab = build_vocab(fra_sentences)
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

# 模型参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# 加载训练好的权重
model.load_state_dict(torch.load("checkpoints/transformer_toy.pth", map_location=device))
model.eval()

# greedy 解码函数
def greedy_decode(model, src_sentence, max_len=20):
    src_ids = encode(src_sentence, src_vocab).unsqueeze(0).to(device)  # [1, src_len]
    memory = model.encoder(src_ids)

    ys = torch.tensor([[tgt_vocab["<bos>"]]], dtype=torch.long).to(device)

    for i in range(max_len):
        out = model.decoder(ys, memory)
        logits = model.generator(out[:, -1])  # 取最后一个词的输出
        prob = F.softmax(logits, dim=-1)
        next_word = torch.argmax(prob, dim=-1).item()

        ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long).to(device)], dim=1)

        if next_word == tgt_vocab["<eos>"]:
            break

    return " ".join([inv_tgt_vocab[idx] for idx in ys[0].tolist()[1:-1]])  # 去掉 <bos> <eos>

# 示例输入
while True:
    src_input = input("\nEnter English sentence (or 'quit'): ").strip().lower()
    if src_input == "quit":
        break
    print("→ French:", greedy_decode(model, src_input))
