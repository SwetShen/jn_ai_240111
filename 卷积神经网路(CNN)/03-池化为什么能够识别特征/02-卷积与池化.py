from torchsummary import summary
from torch import nn

# 卷积
layer1 = nn.Sequential(
    nn.Conv2d(3, 3, 3, 1, 1),
    nn.Conv2d(3, 3, 3, 2, 1)
)

# 池化
layer2 = nn.Sequential(
    nn.Conv2d(3, 3, 3, 1, 1),
    # MaxPool2d 在2x2区域中找到一个最大值输出
    nn.MaxPool2d(2, 2)
)

summary(layer1, (3, 28, 28), device="cpu")
print("=" * 30)
summary(layer2, (3, 28, 28), device="cpu")
