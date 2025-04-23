import torch
from torch import nn


class RNNOne2One(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.RNNCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    # x (N, input_size)
    # h (N, hidden_size)
    def forward(self, x, h=None):
        N, input_size = x.shape
        if h is None:
            h = torch.zeros(N, self.hidden_size)
        h = self.cell(x, h)
        y = self.fc_out(h)
        return y, h


if __name__ == '__main__':
    model = RNNOne2One(2, 10)
    x = torch.rand(5, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
