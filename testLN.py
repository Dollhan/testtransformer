"""
输入内容
时间：2024/7/30 10:19
"""
from LN import LayerNorm
import torch


# 创建 LayerNorm 实例
ln = LayerNorm(features=512)

# 创建一些虚拟输入数据
x = torch.randn(32, 10, 512)  # 假设批量大小为 32，序列长度为 10，特征维度为 512

# 前向传播
output = ln(x)

# 输出形状应为 (32, 10, 512)
print(output.shape)
