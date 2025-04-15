from torch import nn
import torch
from torchsummary import summary

# 残差块有两种：1、BasicBlock 2、Bottleneck（瓶颈层）
image = torch.randn(1, 32, 112, 112)
# ==================== BasicBlock =============================
layer1 = nn.Sequential(
    # 卷积 + RELU + BN ==> CRB结构
    nn.Conv2d(32, 32, 3, 1, 1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, 3, 1, 1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
)
# 残差块原理
result = layer1(image)
result += image  # 输入，卷积输出的形状一致
print(result.shape)
# ==================== Bottleneck =============================
layer2 = nn.Sequential(
    # 卷积 + RELU + BN ==> CRB结构
    nn.Conv2d(32, 8, 1),
    nn.ReLU(),
    nn.BatchNorm2d(8),
    nn.Conv2d(8, 8, 3, 1, 1),
    nn.ReLU(),
    nn.BatchNorm2d(8),
    nn.Conv2d(8, 32, 1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
)
result = layer2(image)
result += image  # 输入，卷积输出的形状一致
print(result.shape)

summary(layer1, (32, 112, 112), device="cpu")
summary(layer2, (32, 112, 112), device="cpu")
