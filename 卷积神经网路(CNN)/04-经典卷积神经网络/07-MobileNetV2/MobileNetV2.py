from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchsummary import summary
from Bottleneck import Bottleneck


def _make_layers(block, in_channels, t, c, n, s):
    down_sampling = False
    if s == 2:
        down_sampling = True
    return nn.Sequential(*[block(in_channels, c, t, down_sampling) if i == 0
                           else block(c, c, t, False) for i in range(n)])


class MobileNetV2(nn.Sequential):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = Conv2dNormActivation(3, 32,
                                          3, 2, 1,
                                          norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.ReLU6)

        self.block1 = _make_layers(Bottleneck, 32, 1, 16, 1, 1)
        self.block2 = _make_layers(Bottleneck, 16, 6, 24, 2, 2)
        self.block3 = _make_layers(Bottleneck, 24, 6, 32, 3, 2)
        self.block4 = _make_layers(Bottleneck, 32, 6, 64, 4, 2)
        self.block5 = _make_layers(Bottleneck, 64, 6, 96, 3, 1)
        self.block6 = _make_layers(Bottleneck, 96, 6, 160, 3, 2)
        self.block7 = _make_layers(Bottleneck, 160, 6, 320, 1, 1)

        self.conv2 = Conv2dNormActivation(320, 1280,
                                          1, 1,
                                          norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.ReLU6)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = Conv2dNormActivation(1280, num_classes,
                                          1, 1,
                                          norm_layer=None,
                                          activation_layer=None)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        return self.softmax(x)


if __name__ == '__main__':
    model = MobileNetV2()
    summary(model, (3, 224, 224), device="cpu")
