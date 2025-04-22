import torch
import torch.nn as nn

# pytorch 官方 RNN 是一个多对多结构的 RNN
# 官方 API: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#rnn
# num_layers: 不同权重的RNN堆叠多少层
# nonlinearity: 非线性激活函数
# bias: 是否学习偏置
# batch_first: 是否批次数，放到维度的首位
# dropout: dropout 操作，防止过拟合
# bidirectional: 是否是双向RNN
# 前两个参数为 input_size, hidden_size
model = nn.RNN(
    input_size=2,
    hidden_size=10,
    num_layers=3,
    nonlinearity='relu',
    bias=False,
    batch_first=True,
    bidirectional=True
)

# 输入输出形状解释:
# 输入: input, h_0
# input: 输入参数，当只有一个批次数据时 (L, H_in)，否则为 (L, N, H_in)，当 batch_first=True 时，N 在最前面 (N, L, H_in)
# h_0: 输入隐藏状态，当只有一个批次数据时 (D * num_layers, H_out)，否则为 (D * num_layers, N, H_out)
# 输出: output, h_n
# output: 输出参数，当只有一个批次数据时 (L, D * H_out)，否则为 (L, N, D * H_out)
# h_n: 输出隐藏状态，当只有一个批次数据时 (D * num_layers, H_out)，否则为 (D * num_layers, N, H_out)

# 对上述符号的解释如下:
# N: 批次数
# L: 序列长度
# D: 双向RNN时为2，否则为1
# H_in: 序列中，每个输入的长度
# H_out: 隐藏状态的长度


x = torch.rand(5, 3, 2)
h = torch.rand(6, 5, 10)
y, h = model(x, h)
print(y.shape)
print(h.shape)

# 官方的 RNNCell
cell = nn.RNNCell(
    input_size=2,
    hidden_size=10,
    bias=False,
    nonlinearity='relu'
)
