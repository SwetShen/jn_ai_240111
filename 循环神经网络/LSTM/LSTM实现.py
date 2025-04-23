import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 遗忘门全连接
        self.fc_f = nn.Linear(input_size + hidden_size, hidden_size)
        # 输入门全连接
        self.fc_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.fc_c = nn.Linear(input_size + hidden_size, hidden_size)
        # 输出们全连接
        self.fc_o = nn.Linear(input_size + hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    # x (N, input_size)
    # c (N, hidden_size)
    # h (N, hidden_size)
    def forward(self, x, c, h):
        # 连接 x 和 h
        # xh (N, input_size + hidden_size)
        xh = torch.concat((x, h), dim=1)
        # 遗忘门
        ft = self.sigmoid(self.fc_f(xh))
        # 输入门
        it = self.sigmoid(self.fc_i(xh))
        _Ct = self.tanh(self.fc_c(xh))
        # 更新长期记忆
        c = ft * c + it * _Ct
        # 输出门
        ot = self.sigmoid(self.fc_o(xh))
        # 更新短期记忆
        h = ot * self.tanh(c)
        return c, h


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    # x (N, L, input_size)
    # c (N, hidden_size)
    # h (N, hidden_size)
    def forward(self, x, c=None, h=None):
        N, L, input_size = x.shape
        # 初始化长短期记忆
        if c is None:
            c = torch.zeros(N, self.hidden_size)
        if h is None:
            h = torch.zeros(N, self.hidden_size)

        y = []

        # 循环序列
        for i in range(L):
            # 更新长短期记忆
            c, h = self.cell(x[:, i], c, h)
            out = self.fc_out(h)
            y.append(out)

        # 堆叠
        y = torch.stack(y, dim=1)
        return y, c, h


if __name__ == '__main__':
    model = LSTM(2, 10)
    x = torch.rand(5, 3, 2)
    y, c, h = model(x)
    print(y.shape, c.shape, h.shape)
