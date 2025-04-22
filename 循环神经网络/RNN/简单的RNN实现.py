import torch
from torch import nn


class RNNCell(nn.Module):
    # input_size: 输入数据的特征个数
    # hidden_size: 隐藏状态的长度
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 隐藏状态的权重
        self.w_h = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        # 输入数据的权重
        self.w_i = nn.Parameter(torch.randn(input_size, hidden_size), requires_grad=True)
        self.tanh = nn.Tanh()

    # h   (N,           hidden_size): N: 批次数
    # w_h (hidden_size, hidden_size)
    # x   (N,          input_size)
    # w_i (input_size, hidden_size)
    def forward(self, x, h):
        h = self.tanh(h @ self.w_h + x @ self.w_i)
        return h


# 多对多RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size)
        self.w_o = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)

    # x 输入序列，形状 (N, L, input_size): L 序列长度
    # h 隐藏状态 形状 (N, hidden_size)
    def forward(self, x, h=None):
        N, L, input_size = x.shape
        if h is None:
            # 初始化隐藏状态
            h = torch.zeros(N, self.hidden_size)

        outputs = []

        # 循环序列
        for i in range(L):
            # 调用 cell 更新隐藏状态
            h = self.cell(x[:, i], h)
            # 输出
            outputs.append(h @ self.w_o)

        y = torch.stack(outputs, dim=1)
        return y, h


if __name__ == '__main__':
    model = RNN(2, 10)
    x = torch.rand(5, 4, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)

    l = nn.Linear(10, 20)
    print(l.weight.shape)
    print(l.bias.shape)
