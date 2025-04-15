import torch
from torch import nn
from torchsummary import summary

layers1 = nn.Sequential(
    nn.Conv2d(6, 6, 7)
)

layers2 = nn.Sequential(
    nn.Conv2d(6, 6, 3),
    nn.Conv2d(6, 6, 3),
    nn.Conv2d(6, 6, 3)
)

summary(layers1, (6, 7, 7), device="cpu")
print("=" * 50)
summary(layers2, (6, 7, 7), device="cpu")
