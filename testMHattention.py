"""
输入内容
时间：2024/7/29 17:08
"""
import torch
from MHattention import MultiHeadAttention


# 示例数据
batch_size = 2
seq_len = 10
model_dim = 512
num_heads = 8
dropout = 0.1

# 创建 MultiHeadAttention 实例
multihead_attn = MultiHeadAttention(model_dim, num_heads, dropout)

# 随机生成示例数据
key = torch.randn(batch_size, seq_len, model_dim)
value = torch.randn(batch_size, seq_len, model_dim)
query = torch.randn(batch_size, seq_len, model_dim)

# 可选：生成一个注意力掩码，在解码器中，每个位置只能注意到前面的位置，不能注意到后面的位置，以确保模型不会看到未来的信息。
attn_mask = (torch.ones(seq_len, seq_len)
             .tril().unsqueeze(0)
             .repeat(batch_size, 1, 1))

# 前向传播
output, attention_weights = multihead_attn(key, value, query, attn_mask)

# 输出结果
print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)
