from torch import nn
from torchsummary import summary

# 正常的卷积
layer1 = nn.Sequential(
    nn.Conv2d(6, 6, 3, 1, 1)
)

# 深度卷积
# groups属性会将每个输入通道单独进行卷积操作
# 一般深度的输入通道数与输出通道数保持一致
layer2 = nn.Sequential(
    nn.Conv2d(6, 6, 3, 1, 1, groups=6)
)

summary(layer1, (6, 20, 20), device="cpu")
print("=" * 40)
summary(layer2, (6, 20, 20), device="cpu")
