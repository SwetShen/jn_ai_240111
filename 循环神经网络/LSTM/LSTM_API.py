import torch
import torch.nn as nn

x = torch.rand(5, 3, 2)

model = nn.LSTM(
    input_size=2,
    hidden_size=10,
    num_layers=2,
    bias=True,
    batch_first=True,
    bidirectional=True,
    # 将 LSTM 输出进行投影(project)，投影就是全连接
    proj_size=6
)

y, (h, c) = model(x)
print(y.shape, h.shape, c.shape)

cell = nn.LSTMCell(
    input_size=2,
    hidden_size=10,
    bias=True,
)
