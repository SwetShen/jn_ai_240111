from torch import nn
from torchsummary import summary
import torch

net = nn.Sequential(
    nn.Conv2d(6, 6, 3, 1, 1),
    nn.BatchNorm2d(6)  # 6个w与6个b， 特别注意batch_norm 虽然是归一化，但是是有参数的
)

# summary(net, (6, 10, 10), device="cpu")

# 批归一的特点，将原本较小值在0-1变大，将原本的较大值在0-1之间变小。（原因：导致BatchNorm2d加w,b的原因）
# 批归一化的值被压制在-1~1 之间，防止梯度累加导致梯度爆炸或者梯度(每一个卷积后重新批归一)消失
# image = torch.randn(1, 6, 10, 10)
# result = net(image)
# print(result)
