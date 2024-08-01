"""
ScaledDotProductAttention，缩放点积注意力机制的实现
时间：2024/7/29 15:21
"""
import torch
import torch.nn as nn
import numpy as np


# 继承nn.moudule类
class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism

    初始化一个ScaledDotProductAttention实例，其中包含两个成员变量

    dropout：一个nn.Dropout层，用于实现Dropout操作，防止过拟合

    softmax：一个nn.Softmax层，用于计算注意力分布
    """

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
                q: Queries张量，形状为[B, L_q, D_q]，其中 B 是批量大小，
                L_q 是查询序列长度，D_q 是查询向量维度
                k: Keys张量，形状为[B, L_k, D_k]，其中 L_k 是键序列长度，D_k 是键向量维度
                v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
                scale: 缩放因子，一个浮点标量，通常等于 sqrt(D_k)
                attn_mask: Masking张量，形状为[B, L_q, L_k]，用于屏蔽不需要关注的位置。

        Returns:
                上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        # if attn_mask:     # 修改
        if attn_mask is not None:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention
