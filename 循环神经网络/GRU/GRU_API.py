import torch.nn as nn
import torch

x = torch.randn(5, 3, 2)

model = nn.GRU(
    input_size=2,
    hidden_size=10,
    num_layers=3,
    bias=True,
    batch_first=True,
    bidirectional=True,
)

y, h = model(x)
print(y.shape)
print(h.shape)

cell = nn.GRUCell(
    input_size=2,
    hidden_size=10,
    bias=True
)
