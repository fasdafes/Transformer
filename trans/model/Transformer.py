import torch.nn as nn
from .TransformerEncoder import TransformerEncoder
from .TransformerDecoder import TransformerDecoder
from trans.model.EmbeddingWithPE import EmbeddingWithPE


class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            tgt_vocab_size,
            embed_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            ff_hidden_dim = None,
            dropout=0.1,
            max_len=5000
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size = vocab_size,
            embed_dim = embed_dim,
            num_layers = num_encoder_layers,
            num_heads = num_heads,
            ff_hidden_dim = ff_hidden_dim,
            dropout= dropout,
            max_len=max_len
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout,
            max_len=max_len
        )

        self.generator = nn.Linear(embed_dim,tgt_vocab_size)

    def forward(self, src, tgt):
        memory = self.encoder(src)  #
        out = self.decoder(tgt, memory)  #
        logits = self.generator(out)
        return logits

    """
                src → Embedding + PosEnc
                        ↓
                  TransformerEncoder
                        ↓
               encoder_output: [语义表达]

  tgt → Embedding + PosEnc（<bos> + 生成前的内容）
                        ↓
              ┌──────── DecoderLayer ─────────┐
              │                               │
    Masked Self-Attn  ←  学习目标语言内部关系    │
    Cross-Attn ←  访问 encoder 输出，找对齐词汇  │
    FFN + 残差 + LN                           │
              └──────────────────────────────┘
                        ↓
                    decoder_output
                        ↓
                Linear(embed_dim → vocab_size)
                        ↓
                  logits（对词表的预测）
                        ↓
          与 tgt_output（右移一位）比 → Loss
          
          
          | 标记      | 全称                | 作用               |
| ------- | ----------------- | -------------------            |
| `<bos>` | Begin Of Sentence | 提示 decoder 从哪开始            |
| `<eos>` | End Of Sentence   | 教模型“生成结束的信号”，模型学会停止 |

    """