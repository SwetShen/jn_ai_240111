from torch import nn
import torch
from torchsummary import summary

# 在原论文中下采样的虚线结构
image = torch.randn(1, 32, 112, 112)
# ==================== BasicBlock =============================
layer1 = nn.Sequential(
    # 卷积 + RELU + BN ==> CRB结构
    nn.Conv2d(32, 32, 3, 2, 1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, 3, 1, 1),
    nn.ReLU()
)
# ==================== down-samping 下采样 =============================
# 下采样结构不可以使用激活函数
down_samping = nn.Sequential(
    nn.Conv2d(32, 32, 3, 2, 1),
)
# 残差块原理
result1 = layer1(image)
result2 = down_samping(image)
result1 += result2  # 输入，卷积输出的形状一致
# 当上述结果相加完成后，会采用批归一化
print(result1.shape)
