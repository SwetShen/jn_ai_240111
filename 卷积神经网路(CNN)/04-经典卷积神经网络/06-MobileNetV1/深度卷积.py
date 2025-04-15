from torch import nn
import torch
from torchsummary import summary

# 普通卷积
conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=False)

# 深度卷积
# groups=3，将每一个卷积核分配到一个通道中（分到一组）
conv2 = nn.Conv2d(3, 3, 3, 1, 1, groups=3, bias=False)

summary(nn.Sequential(conv1), (3, 5, 5), device="cpu")  # 81
summary(nn.Sequential(conv2), (3, 5, 5), device="cpu")  # 27
