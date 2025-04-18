from torch import nn
from torchvision.ops import Conv2dNormActivation


# 倒置残差块
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, t, c, s):
        super().__init__()
        # 扩展卷积
        self.exp = Conv2dNormActivation(in_channels, in_channels * t, 1, activation_layer=nn.ReLU6, stride=s)
        # 深度卷积
        self.depthwise = Conv2dNormActivation(in_channels * t, in_channels * t, 3, groups=in_channels * t,
                                              activation_layer=nn.ReLU6)
        # 逐点卷积
        # 残差最后层逐点卷积不使用非线性激活
        self.pointwise = Conv2dNormActivation(in_channels * t, c, 1, activation_layer=None)

        # 是否进行残差
        self.is_res = in_channels == c and s == 1

    def forward(self, x):
        # 恒等映射
        identity = x
        # 扩展卷积
        x = self.exp(x)
        # 深度卷积
        x = self.depthwise(x)
        # 逐点卷积
        x = self.pointwise(x)
        # 残差
        if self.is_res:
            x = identity + x
        # MNV2 倒置残差的最后采用线性瓶颈，所以残差后也不激活
        return x


# 倒置残差层，其中 倒置残差块将重复 n 次
class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, t, c, n, s):
        super().__init__()
        self.stack = nn.Sequential(
            # 第一个倒置残差块输入通道和下采样，单独处理
            InvertedResidualBlock(in_channels, t, c, s),
            *(InvertedResidualBlock(c, t, c, 1) for _ in range(n - 1))
        )

    def forward(self, x):
        return self.stack(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.c1 = Conv2dNormActivation(3, 32, 3, stride=2, activation_layer=nn.ReLU6)
        self.bottleneck1 = BottleneckLayer(32, 1, 16, 1, 1)
        self.bottleneck2 = BottleneckLayer(16, 6, 24, 2, 2)
        self.bottleneck3 = BottleneckLayer(24, 6, 32, 3, 2)
        self.bottleneck4 = BottleneckLayer(32, 6, 64, 4, 2)
        self.bottleneck5 = BottleneckLayer(64, 6, 96, 3, 1)
        self.bottleneck6 = BottleneckLayer(96, 6, 160, 3, 2)
        self.bottleneck7 = BottleneckLayer(160, 6, 320, 1, 1)
        self.c2 = Conv2dNormActivation(320, 1280, 1, activation_layer=nn.ReLU6)
        self.avg_pool = nn.AvgPool2d(7)
        self.c3 = Conv2dNormActivation(1280, num_classes, 1, activation_layer=None)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.c1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.c2(x)
        x = self.avg_pool(x)
        x = self.c3(x)
        # Nx1000x1x1
        y = self.log_softmax(x.reshape(x.shape[0], -1))
        return y


if __name__ == '__main__':
    import torch
    model = MobileNetV2()
    x = torch.rand(5, 3, 224, 224)
    print(model(x).shape)
    # p.numel(): 求张量的元素数量
    print(sum([p.numel() for p in model.parameters()]))
