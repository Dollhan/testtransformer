"""
输入内容
时间：2024/7/29 9:08
"""
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  #输出为True，则安装无误

import torch

# 创建张量
scalar = torch.tensor(3)  # 标量
vector = torch.tensor([1, 2, 3])  # 向量
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 矩阵
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 三维张量

# 打印张量及其属性
print("Scalar:", scalar)
print("Scalar Shape:", scalar.shape)

print("\nVector:", vector)
print("Vector Shape:", vector.shape)

print("\nMatrix:\n", matrix)
print("Matrix Shape:", matrix.shape)

print("\n3D Tensor:\n", tensor_3d)
print("3D Tensor Shape:", tensor_3d.shape)

# 张量操作示例
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 加法
c = a + b
print("\nAddition of a and b:", c)

# 乘法
d = a * b
print("Multiplication of a and b:", d)

# 矩阵乘法
e = torch.matmul(matrix, matrix.T)
print("\nMatrix Multiplication:\n", e)
