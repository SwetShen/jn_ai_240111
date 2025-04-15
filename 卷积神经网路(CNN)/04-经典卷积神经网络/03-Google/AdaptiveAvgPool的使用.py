import torch
from torch import nn

layer = nn.Sequential(
    nn.Conv2d(6, 6, 3, 1, 1),
    nn.AdaptiveAvgPool2d(1)  # 7 / 1
)

image = torch.rand(1, 6, 7, 7)
result = layer(image)

print(result.shape)  # torch.Size([1, 6, 1, 1])
