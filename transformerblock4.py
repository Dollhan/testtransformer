"""
输入内容
时间：2024/7/31 下午2:38
"""
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 定义 Q, K, V 的权重矩阵
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # 输出线性层
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 将 Q, K, V 分成多头
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        weighted_value = torch.matmul(attention_weights, value)

        # 拼接多头
        weighted_value = weighted_value.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出线性变换
        output = self.out_linear(weighted_value)

        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_()
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力层
        attn_output, _ = self.self_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        # 前馈网络
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout(ff_output)
        out2 = self.layer_norm2(out1 + ff_output)

        return out2

# 示例
d_model = 512  # 特征维度
num_heads = 8  # 头数
d_ff = 2048  # 前馈网络的隐藏层维度
dropout = 0.1

encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
positional_encoding = PositionalEncoding(d_model)

input_data = torch.randn(1, 3, d_model)  # 假设输入序列长度为10
print("input", input_data)
encoded_data = positional_encoding(input_data)
output = encoder_layer(encoded_data, None)
print("output", output)
print("Output shape:", output.shape)