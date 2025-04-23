import torch
from torch import nn


class RNNMany2One(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.RNNCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    # x (N, L, input_size)
    # h (N, hidden_size)
    def forward(self, x, h=None):
        N, L, input_size = x.shape
        if h is None:
            h = torch.zeros(N, self.hidden_size)
        # 循环编码输入序列
        for i in range(L):
            h = self.cell(x[:, i], h)
        # 最后输出一个值
        y = self.fc_out(h)
        return y, h


if __name__ == '__main__':
    model = RNNMany2One(2, 10)
    x = torch.rand(5, 3, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
