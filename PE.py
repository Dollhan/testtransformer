"""
Positional encoding的实现
时间：2024/7/30 13:45
"""
import torch
import numpy as np
import torch.nn as nn

# 定义 PositionalEncoding 类
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        # 构造位置编码矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        # 创建一个全0的 pad_row，表示 `<PAD>` 的位置编码
        pad_row = torch.zeros([1, d_model])

        # 将 pad_row 与 position_encoding 拼接
        position_encoding = torch.cat((pad_row, torch.from_numpy(position_encoding)))

        # 创建嵌入操作
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):
        """
        :param input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。
        :return:返回这一批序列的位置编码，进行了对齐。
        """
        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)


# # 示例代码
# if __name__ == '__main__':
#     d_model = 3
#     max_seq_len = 10
#     pe_layer = PositionalEncoding(d_model, max_seq_len)
#
#     # 创建一个示例输入
#     batch_size = 3
#     input_lengths = torch.tensor([[5], [8], [3]])
#
#     # 调用 forward 方法
#     positional_encodings = pe_layer(input_lengths)
#
#     print("Positional Encodings Shape:", positional_encodings.shape)
#     print("Positional Encodings:")
#     print(positional_encodings)