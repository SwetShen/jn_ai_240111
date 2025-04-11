import torch
from torch import nn
from torchsummary import summary

# 卷积目的：图像特征进行细化（将特征数量变少）
# 一般自定义卷积，可以将图像缩减到 5，6，7，8 这几种大小即可
# CNN 卷积神经网络 图像分类
# 通道数一般选择16 32 64 128 256 512 ....

# 假设现在输入图像大小为28x28 ---> 5x5或者6x6或者7x7或者8x8

model = nn.Sequential(
    # 输入层
    nn.Conv2d(3, 16, 7),
    nn.ReLU(),  # ReLU 在图像多分类中可以缩短模型的推理时间（relu公式简单）
    # 隐藏层
    nn.Conv2d(16, 32, 5),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3),  # (batch_size,128,8,8)
    nn.ReLU(),
    # 输出层
    nn.Flatten(),  # (batch_size,128x8x8)
    nn.Dropout(),
    nn.Linear(128 * 8 * 8, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10),
    nn.LogSoftmax(dim=-1)
)

if __name__ == '__main__':
    summary(model, (3, 28, 28), device="cpu")
