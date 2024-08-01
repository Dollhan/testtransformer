"""
输入内容
时间：2024/7/29 15:42
"""
import numpy as np
import torch
from SDPattention import ScaledDotProductAttention


# 创建一个ScaledDotProductAttention实例
attention = ScaledDotProductAttention(attention_dropout=0.1)

# 构造输入数据
B, L_q, L_k, D_k, D_v = 2, 3, 4, 5, 6
q = torch.randn(B, L_q, D_k)
k = torch.randn(B, L_k, D_k)
v = torch.randn(B, L_k, D_v)
scale = np.sqrt(D_k)
attn_mask = torch.tensor([
    [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]],
    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
])

# 调用forward方法
context, attention_weights = attention(q, k, v, scale, attn_mask)

# 输出结果
print("Context shape:", context.shape)
print("Attention weights shape:", attention_weights.shape)
