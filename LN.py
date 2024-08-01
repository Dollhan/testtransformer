"""
LayerNorm的实现，pytorch已经实现
时间：2024/7/30 10:06
"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """实现Layer Normalization。实际上，PyTorch已经内置了这个功能，见nn.LayerNorm。"""

    def __init__(self, features, epsilon=1e-6):
        """
        初始化LayerNorm模块。

        参数:
            features (int): 特征的维度，即模型的维度。在Transformer中通常为512。
            epsilon (float): 一个非常小的正数，用于防止除法中出现除以零的情况。
        """
        super(LayerNorm, self).__init__()

        # 初始化可学习的参数 gamma (缩放系数) 和 beta (偏移系数)
        # gamma 和 beta 的初始值分别为全1张量和全0张量
        # 这些参数将在训练过程中被优化
        self.gamma = nn.Parameter(torch.ones(features))  # alpha
        self.beta = nn.Parameter(torch.zeros(features))  # beta
        self.epsilon = epsilon  # 防止除零错误的小数值

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (Tensor): 输入序列张量，形状为 [B, L, D]，其中 B 是批量大小，L 是序列长度，D 是特征维度。

        返回:
            Tensor: 归一化后的输出张量，形状为 [B, L, D]。
        """
        # 计算每个样本的特征维度上的均值
        mean = x.mean(dim=-1, keepdim=True)  # dim=-1 表示在最后一个维度（即特征维度）上求均值
                                               # keepdim=True 保证输出仍然是[B, L, 1]的形状

        # 计算每个样本的特征维度上的标准差
        std = x.std(dim=-1, keepdim=True)  # dim=-1 表示在最后一个维度上求标准差
                                           # keepdim=True 保证输出仍然是[B, L, 1]的形状

        # 应用Layer Normalization公式
        # (x - mean) / (std + epsilon) 进行归一化
        # gamma * ... + beta 进行缩放和平移
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
