# 双向RNN是在单向RNN的基础上，再增加一个 RNNCell，然后将序列反向输入
# 合并输出和隐藏状态:
# 正反向输出的隐藏状态进行堆叠
# 正反向输出的结果进行连接
import torch
# 双向RNN的优点是，让模型考虑序列的过去和未来的发展趋势。
from torch import nn


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 正向
        self.forward_cell = nn.RNNCell(input_size, hidden_size)
        self.forward_fc = nn.Linear(hidden_size, hidden_size)
        # 反向
        self.backward_cell = nn.RNNCell(input_size, hidden_size)
        self.backward_fc = nn.Linear(hidden_size, hidden_size)

    # x (N, L, input_size)
    # h (2, N, hidden_size)
    def forward(self, x, h=None):
        N, L, input_size = x.shape
        if h is None:
            h = torch.zeros(2, N, self.hidden_size)
        # 正向输出的结果
        forward_y = []
        # 正向的隐藏状态
        forward_h = h[0]
        # 正向循环
        for i in range(L):
            # 更新正向的隐藏状态
            forward_h = self.forward_cell(x[:, i], forward_h)
            # 正向输出
            forward_y.append(self.forward_fc(forward_h))
        # 转换成张量
        forward_y = torch.stack(forward_y, dim=1)

        # 将输入序列反向
        x = x.flip(dims=[1])
        backward_y = []
        backward_h = h[1]
        for i in range(L):
            backward_h = self.backward_cell(x[:, i], backward_h)
            backward_y.append(self.backward_fc(backward_h))
        # (N, L, hidden_size)
        backward_y = torch.stack(backward_y, dim=1)

        # 合并正反向的结果
        h = torch.stack([forward_h, backward_h])
        y = torch.concat([forward_y, backward_y], dim=-1)
        return y, h


if __name__ == '__main__':
    model = BiRNN(2, 10)
    x = torch.rand(5, 3, 2)
    y, h = model(x)
    print(y.shape, h.shape)
