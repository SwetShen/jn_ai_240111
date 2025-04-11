import torch
from torch import nn

"""
nn.Conv2d 参数介绍
in_channels: 输入通道数
out_channels: 输出通道数（卷积核的个数）
kernel_size: 卷积核的大小
stride: 步长（每次卷积位移的距离） 默认值：1
bias: 偏置（True 有b参数，False 没有b参数） 默认值：True
padding: 填充，默认值：0  格式：1、字符串('same'或者'valid') 2、数值
        'same': 输出输入图像大小保持不变  'valid':不加边距（padding=0）
"""
conv1_layer = nn.Sequential(
    nn.Conv2d(3, 6, 5, 1, 'same')
)

conv2_layer = nn.Sequential(
    nn.Conv2d(3, 6, 3)
)

image = torch.rand(1, 3, 20, 20)

result1 = conv1_layer(image)  # (1,6,20,20)
result2 = conv2_layer(image)  # (1,6,18,18)
print(result1.shape)
print(result2.shape)
