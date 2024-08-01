"""
Mask掩码设计
时间：2024/7/30 10:31
"""
import torch


# 填充掩码（padding mask）
def padding_mask(seq_k, seq_q):
    """
    :param seq_k: 键序列，形状为 [B, L]，其中 B 是批量大小，L 是序列长度。
    :param seq_q: 查询序列，形状同样为 [B, L]。
    :return:一个布尔型张量，形状为 [B, L_q, L_k]，其中 L_q 是 seq_q 的长度，L_k 是 seq_k 的长度。
    如果某个位置 (i, j, k) 的值为 True，则表示 seq_k 中第 k 个位置的元素为填充位。
    """
    len_q = seq_q.size(1)
    # 创建一个与 seq_k 相同大小的布尔张量，其中非填充位为 False，填充位为 True
    # print("len_q", len_q)
    pad_mask = seq_k.eq(0)  # [B, 1, L_k]
    # print("pad_mask", pad_mask)
    # 扩展维度以匹配查询序列的长度
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    # print("pad_mask", pad_mask)
    # print(pad_mask.shape)
    return pad_mask


def sequence_mask(seq):
    """
    :param seq: 序列张量，形状为 [B, L]。
    :return:torch.Tensor: 布尔型张量，形状为 [B, L, L]。对于每个批次中的序列，
    该掩码会创建一个上三角矩阵，其中主对角线之后的所有元素都被标记为 True。
    """
    batch_size, seq_len = seq.size()
    # 创建一个下三角矩阵，其中非对角线元素为 True，对角线及其以下元素为 False
    sequence_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    # 将下三角矩阵扩展到每个批次
    sequence_mask = sequence_mask.unsqueeze(0).expand(batch_size, -1, -1)
    return sequence_mask
