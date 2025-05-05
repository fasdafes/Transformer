import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import dtype


def generate_causal_mask(seq_len,device=None):
    #生成下三角 mask，用于 decoder 中的 masked self-attention。
    #True 表示可见，False 表示 masked 掉。
    mask = torch.tril(torch.ones((seq_len,seq_len),dtype=torch.bool))
    # torchones 做成一个seq_len*seq_len的矩阵，dtype=bool强制转化成boolean类型
    # 取一个下三角形，只保留对角线和坐下部分
    #[[True, False, False, False, False],
     #[True, True, False, False, False],
     #[True, True, True, False, False],
     #[True, True, True, True, False],
     #[True, True, True, True, True]]
    if device is not None:
        mask = mask.to(device)
    return mask #shape：（seq_Len,seq_len）


