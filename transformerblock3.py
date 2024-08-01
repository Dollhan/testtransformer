"""
通过对每个模块进行单独测试，并打印中间结果来观察输入和输出的变化。
我们可以分阶段逐步对每个模块进行操作，从而清晰地看到每一步对数据的处理效果。
时间：2024/7/31 上午8:45
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(SimpleTransformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = EmbeddingLayer(vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.word_embedding(x)
        print("Embedding Output:\n", out)

        out = self.position_embedding(out)
        print("Positional Encoding Output:\n", out)

        for i, layer in enumerate(self.layers):
            out = layer(out, out, out, mask)
            print(f"Transformer Block {i + 1} Output:\n", out)

        out = self.fc_out(out)
        return out


# Hyperparameters
vocab_size = 10000
embed_size = 512
num_layers = 2  # 这里只用2层，以便简化输出
heads = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
forward_expansion = 4
dropout = 0.1
max_length = 100

# Example input (batch_size = 1, sequence_length = 10)
x = torch.randint(0, vocab_size, (1, 10)).to(device)
mask = None  # Add mask here if needed

# Initialize and run the model
model = SimpleTransformer(vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length).to(device)
out = model(x, mask)
print("Final Output:\n", out.shape)  # Expected shape (1, 10, vocab_size)
