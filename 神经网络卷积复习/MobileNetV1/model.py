import math

from torch import nn
from torchvision.ops import Conv2dNormActivation


# 深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = Conv2dNormActivation(in_channels, in_channels, 3, stride=stride, groups=in_channels)
        self.pointwise = Conv2dNormActivation(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.c1 = Conv2dNormActivation(3, 32, 3, stride=2)
        self.dsc1 = DepthwiseSeparableConv(32, 64, stride=1)
        self.dsc2 = DepthwiseSeparableConv(64, 128, stride=2)
        self.dsc3 = DepthwiseSeparableConv(128, 128, stride=1)
        self.dsc4 = DepthwiseSeparableConv(128, 256, stride=2)
        self.dsc5 = DepthwiseSeparableConv(256, 256, stride=1)
        self.dsc6 = DepthwiseSeparableConv(256, 512, stride=2)
        self.dsc7 = nn.Sequential(
            *(DepthwiseSeparableConv(512, 512, stride=1) for _ in range(5))
        )
        self.dsc8 = DepthwiseSeparableConv(512, 1024, stride=2)
        # 输入 dsc9 时，图片的大小
        input_size = 320 / 32
        p = math.ceil((input_size + 1) / 2)
        self.dsc9 = nn.Sequential(
            Conv2dNormActivation(1024, 1024, 3, stride=2, padding=p, groups=1024),
            Conv2dNormActivation(1024, 1024, 1)
        )
        # self.avg_pool = nn.AvgPool2d(7)
        # 自适应池化: 池化核大小会自动调整，只需要填输出形状即可
        # 参数 1 代表输出图像大小为 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # Nx3x224x224
        x = self.c1(x)
        # Nx32x112x112
        x = self.dsc1(x)
        # Nx64x112x112
        x = self.dsc2(x)
        # Nx128x56x56
        x = self.dsc3(x)
        # Nx128x56x56
        x = self.dsc4(x)
        # Nx256x28x28
        x = self.dsc5(x)
        # Nx256x28x28
        x = self.dsc6(x)
        # Nx512x14x14
        x = self.dsc7(x)
        # Nx512x14x14
        x = self.dsc8(x)
        # Nx1024x7x7
        x = self.dsc9(x)
        # Nx1024x7x7
        x = self.avg_pool(x)
        # Nx1024x1x1
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    import torch

    model = MobileNetV1()
    # x = torch.rand(5, 3, 224, 224)
    x = torch.rand(5, 3, 320, 320)
    print(model(x).shape)
