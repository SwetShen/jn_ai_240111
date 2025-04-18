from torch import nn
from torchvision.ops import Conv2dNormActivation


# 基础残差块
class BasicBlock(nn.Module):
    # is_downsample: 是否下采样，当输入图像和输出图像不一样大时，代表下采样为 True
    def __init__(self, in_channels, out_channels, is_downsample=False):
        super().__init__()
        # 若下采样，则第一个卷积步幅为 2
        self.c1 = Conv2dNormActivation(in_channels, out_channels, 3, activation_layer=nn.ReLU,
                                       norm_layer=nn.BatchNorm2d, stride=2 if is_downsample else 1)
        # 残差块的最后一个卷积层，不激活
        self.c2 = Conv2dNormActivation(out_channels, out_channels, 3, activation_layer=None, norm_layer=nn.BatchNorm2d)

        # 下采样卷积
        # 1. 当输入通道和输出通道不等时，则通过一个卷积，让通道数相等
        # 2. 当 is_downsample 为 True 时，则进行下采样
        self.downsample_conv = Conv2dNormActivation(in_channels, out_channels, 1,
                                                    activation_layer=None,
                                                    stride=2 if is_downsample else 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # 恒等映射
        identity = x
        x = self.c1(x)
        x = self.c2(x)
        # 需要卷积统一通道数
        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
        # 相加 链接
        # x += identity  # 就地运算（in-place），反向传播时会报错
        x = x + identity
        # 激活
        x = self.relu(x)
        return x


# 瓶颈块
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, is_downsample=False):
        super().__init__()
        # 输出通道的四分之一
        _d_4_dim = out_channels // 4
        # 第一层卷积要考虑下采样
        self.c1 = Conv2dNormActivation(in_channels, _d_4_dim, 1, stride=2 if is_downsample else 1)
        self.c2 = Conv2dNormActivation(_d_4_dim, _d_4_dim, 3)
        # 最后一层卷积不激活
        self.c3 = Conv2dNormActivation(_d_4_dim, out_channels, 1, activation_layer=None)
        self.downsample_conv = Conv2dNormActivation(in_channels, out_channels, 1, stride=2 if is_downsample else 1,
                                                    activation_layer=None) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # 恒等映射
        identity = x
        # 3层卷积
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
        # 残差连接
        x = x + identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    # block: 残差块的类型
    # config: 残差块堆叠次数的配置，是一个数字列表
    def __init__(self, block, config=[2, 2, 2, 2], num_classes=1000):
        super().__init__()
        self.c1 = Conv2dNormActivation(3, 64, 7, stride=2)
        # 根据 block 的类别构造输出通道数列表
        out_channels = [64, 128, 256, 512] if block == BasicBlock else \
            [256, 512, 1024, 2048]

        self.block_list = nn.ModuleList([])
        # 第一个残差组合的处理
        self.block_list.append(
            nn.Sequential(
                block(64, out_channels[0]),
                *(block(out_channels[0], out_channels[0]) for _ in range(config[0] - 1))
            )
        )

        # 循环 4 个残差块组合
        for i, cfg in enumerate(config[1:]):
            self.block_list.append(nn.Sequential(
                # 第一个残差块需要下采样
                block(out_channels[i], out_channels[i + 1], is_downsample=True),
                # 循环 cfg - 1 个残差块
                *(block(out_channels[i + 1], out_channels[i + 1]) for _ in range(cfg - 1))
            ))

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.classifier = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(start_dim=1),
            nn.Linear(out_channels[-1], num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # Nx3x224x224
        x = self.c1(x)
        # Nx64x112x112
        x = self.max_pool(x)
        # Nx64x56x56
        for block in self.block_list:
            x = block(x)
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    import torch

    # bb = BasicBlock(64, 128, is_downsample=True)
    # bn = Bottleneck(64, 128, is_downsample=True)
    # x = torch.rand(5, 64, 32, 32)
    # print(bb(x).shape)
    # print(bn(x).shape)

    # resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3])
    x = torch.rand(5, 3, 224, 224)
    print(resnet152(x).shape)
