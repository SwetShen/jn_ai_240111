from torch import nn
from torchsummary import summary

net = nn.Sequential(
    nn.Conv2d(6, 6, 3, 1, 1),
    nn.BatchNorm2d(6)  # 6个w与6个b， 特别注意batch_norm 虽然是归一化，但是是有参数的
)

summary(net, (6, 10, 10), device="cpu")
