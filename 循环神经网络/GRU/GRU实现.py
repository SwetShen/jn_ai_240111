import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 更新门权重
        self.Wz = nn.Parameter(torch.randn(input_size, hidden_size), requires_grad=True)
        self.Uz = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        # 重置门权重
        self.Wr = nn.Parameter(torch.randn(input_size, hidden_size), requires_grad=True)
        self.Ur = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        # 候选隐藏状态的权重
        self.W = nn.Parameter(torch.randn(input_size, hidden_size), requires_grad=True)
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    # x (N, input_size)
    # h (N, hidden_size)
    def forward(self, x, h):
        # 更新门
        zt = self.sigmoid(x @ self.Wz + h @ self.Uz)
        # 重置门
        rt = self.sigmoid(x @ self.Wr + h @ self.Ur)
        # 计算候选隐藏状态
        _h = self.tanh((h * rt) @ self.U + x @ self.W)
        # 更新隐藏状态
        h = (1 - zt) * _h + zt * h
        return h


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    # x (N, L, input_size)
    # h (N, hidden_size)
    def forward(self, x, h=None):
        N, L, input_size = x.shape
        if h is None:
            h = torch.zeros(N, self.hidden_size)
        y = []
        for i in range(L):
            h = self.cell(x[:, i], h)
            out = self.fc_out(h)
            y.append(out)
        y = torch.stack(y, dim=1)
        return y, h


if __name__ == '__main__':
    model = GRU(2, 10)
    x = torch.rand(5, 3, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
