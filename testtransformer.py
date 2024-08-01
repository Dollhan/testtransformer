"""
输入内容
时间：2024/7/31 下午2:59
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



i = 0
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_q = nn.Linear(model_dim, model_dim)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        batch_size = key.size(0)

        # 线性变换
        key = self.linear_k(key)    # torch.Size([1, 10, 512])  1:batch、10：序列最大长度、512：序列维度
        value = self.linear_v(value)
        query = self.linear_q(query)
        # print("key:", key.shape)
        # 分割头部
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)  # torch.Size([8, 10, 64])，8个head把序列维度分开
        # print("key:", key.shape)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        # 计算注意力权重
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.dim_per_head)

        # if attn_mask is not None:
        #     scores = scores.masked_fill_(attn_mask, -1e9)
        # print("scores:", scores.shape)  # torch.Size([8, 10, 10])   10×10是因为每个q都要和每个key求相似度
        # print(scores)
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)
        # print("attn:", attn.shape)
        # print(attn)
        # 加权平均
        context = torch.bmm(attn, value)    # attn:10×10 ;value：10×512

        # 合并头部
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)
        # print("context:", context.shape)
        # print(context)

        # 线性变换
        output = self.linear_final(context)
        # print("output+residual", (output + residual).shape)
        # print(output + residual)
        return output + residual, attn

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len=8000):
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, ffn_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, ffn_dim)
        self.w_2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ffn_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        context, attn = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attn

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, num_layers, model_dim, num_heads, ffn_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, inputs_len):
        # 输入嵌入
        outputs = self.embedding(inputs)
        outputs = self.dropout(self.positional_encoding(outputs))

        # 构建掩码
        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for layer in self.layers:
            outputs, attn = layer(outputs, self_attention_mask)
            attentions.append(attn)
            # global i
            # i = i + 1
            # print(i)
            # print(outputs.shape)
            # print(attentions)
        return outputs, attentions

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ffn_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim, ffn_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)

    def forward(self, inputs, encoder_outputs, self_attn_mask=None, context_attn_mask=None):
        # 自注意力机制
        context, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)

        # 规范化
        context = self.layer_norm1(context + inputs)

        # 编码器-解码器注意力
        context, context_attn = self.encoder_attention(encoder_outputs, encoder_outputs, context, context_attn_mask)

        # 规范化
        context = self.layer_norm2(context + context)

        # 前馈网络
        context = self.feed_forward(context)

        # 规范化
        output = self.layer_norm3(context)

        return output, self_attn, context_attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, num_layers, model_dim, num_heads, ffn_dim, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, inputs_len, encoder_outputs, context_attn_mask):
        # 输入嵌入
        outputs = self.embedding(inputs)
        outputs = self.dropout(self.positional_encoding(outputs))

        # 构建掩码
        self_attention_mask = subsequent_mask(inputs.size(-1)).to(inputs.device)

        self_attns = []
        context_attns = []
        for layer in self.layers:
            outputs, self_attn, context_attn = layer(
                outputs, encoder_outputs, subsequent_mask, context_attn_mask)
            self_attns.append(self_attn)
            context_attns.append(context_attn)

        return outputs, self_attns, context_attns

def padding_mask(seq_q, seq_k):
    # 生成一个掩码矩阵，用于屏蔽掉填充的元素
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0).unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    # print(padding_mask.shape)
    return padding_mask

def subsequent_mask(size):
    # 生成一个掩码矩阵，用于屏蔽掉未来的词
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 src_max_len,
                 tgt_vocab_size,
                 tgt_max_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)

        output, enc_self_attn = self.encoder(src_seq, src_len)

        output, dec_self_attn, ctx_attn = self.decoder(
            tgt_seq, tgt_len, output, context_attn_mask)

        output = self.linear(output)
        output = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, ctx_attn

# 辅助函数
def generate_sequence(batch_size, max_length, vocab_size):
    """生成随机的整数序列，用于模拟句子"""
    return torch.LongTensor(batch_size, max_length).random_(0, vocab_size)
# def generate_sequence(batch_size, max_length, vocab_size):
#     """
#     生成随机的整数序列，用于模拟句子。
#
#     参数:
#     - batch_size: 序列的批处理大小。
#     - max_length: 序列的最大长度。
#     - vocab_size: 词汇表大小。
#
#     返回:
#     - sequences_tensor: 一个二维张量，形状为 (batch_size, max_length)，其中每个序列的实际长度不同。
#     - lengths: 包含每个序列实际长度的一维张量。
#     """
#     sequences = []
#     lengths = []
#     for _ in range(batch_size):
#         # 随机生成每个序列的实际长度
#         length = torch.randint(1, max_length + 1, (1,)).item()
#
#         # 生成随机整数序列
#         sequence = torch.randint(0, vocab_size, (length,))
#
#         # 将序列填充到 max_length
#         padded_sequence = F.pad(sequence, (0, max_length - length), mode='constant', value=0)
#
#         sequences.append(padded_sequence)
#         lengths.append(length)
#
#     # 将序列列表转换为张量
#     sequences_tensor = torch.stack(sequences)
#     lengths_tensor = torch.tensor(lengths)
#
#     return sequences_tensor, lengths_tensor

def generate_lengths(batch_size, max_length):
    """生成随机的序列长度，确保每个序列长度不超过最大长度"""
    return torch.LongTensor(batch_size).random_(1, max_length + 1)

# 定义模型参数
src_vocab_size = 10000  # 源语言词汇表大小
src_max_len = 10       # 源序列最大长度   src_vocab_size与src_max_len相乘代表了表示词量的大小
tgt_vocab_size = 9000  # 目标语言词汇表大小
tgt_max_len = 12       # 目标序列最大长度
num_layers = 6          # 编码器/解码器层数
model_dim = 512         # 模型维度
num_heads = 8           # 头数
ffn_dim = 2048          # 前馈网络中间层维度
dropout = 0.1           # Dropout比率

# 创建数据
batch_size = 1      # 样本数(词的个数)：词的个数每个词都可以从10000×10的词汇表中一一对应
src_seq  = generate_sequence(batch_size, src_max_len, src_vocab_size)
print("源语言序列", src_seq)
src_len = generate_lengths(batch_size, src_max_len)
print("每条源序列的实际长度", src_len)
tgt_seq = generate_sequence(batch_size, tgt_max_len, tgt_vocab_size)
print("目标语言序列", tgt_seq)
tgt_len = generate_lengths(batch_size, src_max_len)
print("每条目标序列的实际长度", tgt_len)

# 初始化模型
transformer = Transformer(src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len,
                          num_layers, model_dim, num_heads, ffn_dim, dropout)


# 进行前向传播
output, enc_self_attn, dec_self_attn, ctx_attn = transformer(src_seq, src_len, tgt_seq, tgt_len)

# 输出验证
print("输出的shape:", output.shape)
print("最终输出:", output)
print("Encoder self-attention shape:", enc_self_attn[-1].shape)
# print("编码器的输出:", enc_self_attn[-1])
print("Decoder self-attention shape:", dec_self_attn[-1].shape)
# print("解码器的输出:", dec_self_attn[-1].shape)
