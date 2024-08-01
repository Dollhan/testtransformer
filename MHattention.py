"""
多头注意力机制的实现
时间：2024/7/29 16:23
"""
import torch
import torch.nn as nn
from SDPattention import ScaledDotProductAttention  # 假设这个模块已定义并可用


def residual(sublayer_fn, x):
    return sublayer_fn + x


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制（Multi-Head Attention）的实现。

    初始化方法 (__init__):

    model_dim: 模型的维度（例如512）。

    num_heads: 多头注意力机制中的头的数量（例如8）。

    dropout: Dropout层的丢弃概率。

    初始化三个线性层 linear_k, linear_v, linear_q 用于将输入的键（Key）、值（Value）和
    查询（Query）张量转换成多头注意力机制所需的维度。

    定义 ScaledDotProductAttention 类实例，用于计算缩放点积注意力（scaled dot-product attention）。
    这里假设 ScaledDotProductAttention 类已经定义好并实现了相应的功能。

    定义最终的线性层 linear_final 用于将多头注意力的结果投影回原始的模型维度。

    定义 nn.Dropout 层用于在训练过程中随机丢弃一部分神经元的输出，防止过拟合。

    定义 nn.LayerNorm 层用于对残差连接后的输出进行归一化处理。
    """

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        # 每个头的维度
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        # 线性层用于转换 Key、Value 和 Query 的维度
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        # 缩放点积注意力实例
        self.dot_product_attention = ScaledDotProductAttention(dropout)

        # 最终的线性层用于将多头注意力的结果投影回原始的模型维度
        self.linear_final = nn.Linear(model_dim, model_dim)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # LayerNorm 层
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        """
        前向传播方法 (forward):

        key: 键张量。

        value: 值张量。

        query: 查询张量。

        attn_mask: 注意力掩码张量（可选）。

        返回多头注意力的结果和注意力权重。
        """

        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # 使用线性层对 Key、Value 和 Query 进行转换
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 将 Key、Value 和 Query 张量按头拆分
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # 如果提供了注意力掩码，则对其进行复制以适应多头注意力的需求
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # 缩放点积注意力计算
        # 注意：这里的 scale 计算有误，应该是 self.dim_per_head 的平方根的倒数
        scale = (dim_per_head ** -0.5)
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # 将各个头的结果拼接在一起
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # 通过最终的线性层进行投影
        output = self.linear_final(context)

        # 应用 Dropout
        output = self.dropout(output)

        # 残差连接后添加 LayerNorm
        output = self.layer_norm(residual + output)

        # 返回多头注意力的结果和注意力权重
        return output, attention
