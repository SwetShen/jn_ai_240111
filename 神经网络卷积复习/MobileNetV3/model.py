from torch import nn
from torchvision.ops import Conv2dNormActivation


# 序列激发块
class SE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2dNormActivation(in_channels, in_channels // 4, 1, activation_layer=nn.ReLU)
        self.fc2 = Conv2dNormActivation(in_channels // 4, in_channels, 1, activation_layer=nn.Hardsigmoid)

    def forward(self, x):
        weight = self.avg_pool(x)
        weight = self.fc1(weight)
        weight = self.fc2(weight)
        return weight * x


# 瓶颈倒置残差块
class Bottleneck(nn.Module):
    # kernel_size: 深度卷积，卷积核大小
    # exp_size: 扩展卷积的输出通道数
    # out: 瓶颈块的输出通道
    # se: 是否序列激发
    # nl: 非线性激活函数
    # s: 步幅
    def __init__(self, in_channels, kernel_size, exp_size, out, se, nl, s):
        super().__init__()
        # 扩展卷积
        self.exp_conv = Conv2dNormActivation(in_channels, exp_size, 1, stride=s, activation_layer=nl)
        # 深度卷积
        self.depth_wise = Conv2dNormActivation(exp_size, exp_size, kernel_size, groups=exp_size, activation_layer=nl)
        # 逐点卷积
        # nn.Identity() 线性激活，也就是 f(x) = x
        self.point_wise = Conv2dNormActivation(exp_size, out, 1, activation_layer=nn.Identity)
        # 是否残差
        # 步幅为一且输入输出通道相等，才进行残差
        self.is_res = s == 1 and in_channels == out
        self.se = SE(exp_size) if se else None

    def forward(self, x):
        # 恒等映射
        identity = x
        x = self.exp_conv(x)
        x = self.depth_wise(x)
        if self.se is not None:
            x = self.se(x)
        x = self.point_wise(x)
        if self.is_res:
            x = x + identity
        return x


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.c1 = Conv2dNormActivation(3, 16, 3, stride=2, activation_layer=nn.Hardswish)
        self.bneck1 = Bottleneck(16, 3, 16, 16, True, nn.ReLU, 2)
        self.bneck2 = Bottleneck(16, 3, 72, 24, False, nn.ReLU, 2)
        self.bneck3 = Bottleneck(24, 3, 88, 24, False, nn.ReLU, 1)
        self.bneck4 = Bottleneck(24, 5, 96, 40, True, nn.Hardswish, 2)
        self.bneck5 = Bottleneck(40, 5, 240, 40, True, nn.Hardswish, 1)
        self.bneck6 = Bottleneck(40, 5, 240, 40, True, nn.Hardswish, 1)
        self.bneck7 = Bottleneck(40, 5, 120, 48, True, nn.Hardswish, 1)
        self.bneck8 = Bottleneck(48, 5, 144, 48, True, nn.Hardswish, 1)
        self.bneck9 = Bottleneck(48, 5, 288, 96, True, nn.Hardswish, 2)
        self.bneck10 = Bottleneck(96, 5, 576, 96, True, nn.Hardswish, 1)
        self.bneck11 = Bottleneck(96, 5, 576, 96, True, nn.Hardswish, 1)
        self.c2 = nn.Sequential(
            Conv2dNormActivation(96, 576, 1, activation_layer=nn.Hardswish),
            SE(576),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            Conv2dNormActivation(576, 1024, 1, activation_layer=nn.Hardswish, norm_layer=None),
            Conv2dNormActivation(1024, num_classes, 1, norm_layer=None),
            # Nx1000x1x1
            nn.Flatten(start_dim=1),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.bneck1(x)
        x = self.bneck2(x)
        x = self.bneck3(x)
        x = self.bneck4(x)
        x = self.bneck5(x)
        x = self.bneck6(x)
        x = self.bneck7(x)
        x = self.bneck8(x)
        x = self.bneck9(x)
        x = self.bneck10(x)
        x = self.bneck11(x)
        x = self.c2(x)
        x = self.avg_pool(x)
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    import torch

    # model = Bottleneck(8, 32, 8, True, nn.ReLU, 1)
    # x = torch.rand(5, 8, 224, 224)
    # print(model(x).shape)

    model = MobileNetV3_Small(1000)
    x = torch.rand(5, 3, 224, 224)
    print(model(x).shape)
