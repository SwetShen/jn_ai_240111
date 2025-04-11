from torch import nn
from torchsummary import summary

"""
nn.Conv2d 参数介绍
in_channels: 输入通道数
out_channels: 输出通道数（卷积核的个数）
kernel_size: 卷积核的大小
stride: 步长（每次卷积位移的距离） 默认值：1
bias: 偏置（True 有b参数，False 没有b参数） 默认值：True
"""
conv_layer = nn.Sequential(
    nn.Conv2d(3, 6, 3, 1, bias=True)
)
# 关于上述的w和b的计算
# w : 卷积核中参数量 3x3x6 = 54
# b : 与卷积核数量一直 6
# 参与运算的总参数量（Param）大小：3x6x3x3 = 162

summary(conv_layer, (3, 20, 20), device="cpu")
