"""
输入内容
时间：2024/7/30 11:13
"""
import torch
from Mask import padding_mask, sequence_mask


# 示例输入
seq_k = torch.tensor([[1, 2, 0], [1, 0, 0]])  # 假设0代表PAD
seq_q = torch.tensor([[1, 2, 3], [1, 2, 0]])  # 假设0代表PAD

# 调用函数
pad_mask = padding_mask(seq_k, seq_q)
seq_mask = sequence_mask(seq_k)

print("Padding Mask:")
print(pad_mask)
print("\nSequence Mask:")
print(seq_mask)
