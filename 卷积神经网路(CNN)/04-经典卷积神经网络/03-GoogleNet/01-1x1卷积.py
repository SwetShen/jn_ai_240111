import torch
from torch import nn
from torchsummary import summary

layer1 = nn.Sequential(
    nn.Conv2d(16, 32, 3, 1, 1),
    nn.Conv2d(32, 32, 3, 1, 1)
)

layer2 = nn.Sequential(
    # 1x1卷积主要用于升维，降维上的功能。（相较于传统的卷积，参数量更少）
    nn.Conv2d(16, 32, 1),  # 参数量减少、提高卷积的速度
    nn.Conv2d(32, 32, 3, 1, 1)
)

summary(layer1, (16, 24, 24), 10, "cpu")
print("==" * 50)
summary(layer2, (16, 24, 24), 10, "cpu")
