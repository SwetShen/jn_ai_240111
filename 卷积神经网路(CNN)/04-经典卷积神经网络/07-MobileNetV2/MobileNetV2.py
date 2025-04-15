from torch import nn
import torch
from torchvision.ops import Conv2dNormActivation
from torchsummary import summary


# Depthwise Separable Convolution 深度可分离卷积
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #  深度卷积
        self.depth_wise = Conv2dNormActivation(in_channels, in_channels, 3,
                                               stride=stride, padding=1, groups=in_channels,
                                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        #  逐点卷积
        self.point_wise = Conv2dNormActivation(in_channels, out_channels, 1,
                                               norm_layer=nn.BatchNorm2d, activation_layer=None)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


#  倒置残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, rate, down_sampling=False):
        super().__init__()

        self.conv1 = Conv2dNormActivation(in_channels, in_channels * rate, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)

        self.conv2 = DSConv(in_channels * rate, out_channels, stride=2 if down_sampling else 1)

        self.res_conv = Conv2dNormActivation(in_channels, out_channels, 1,
                                             stride=2 if down_sampling else 1,
                                             norm_layer=nn.BatchNorm2d, activation_layer=None)

    def forward(self, x):
        identify = x
        x = self.conv1(x)
        x = self.conv2(x)

        identify = self.res_conv(identify)

        x += identify
        return x


def _make_layers(block, in_channels, out_channels, rate, block_num, down_sampling=None):
    return nn.Sequential(*[block(in_channels, out_channels, rate, down_sampling) if i == 0 else
                           block(out_channels, out_channels, rate) for i in range(block_num)])


class MobileNetV2(nn.Module):
    def __init__(self, classes_num=1000):
        super().__init__()
        self.classes_num = classes_num

        self.conv1 = Conv2dNormActivation(3, 32, 3,
                                          stride=2, padding=1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.conv2 = _make_layers(InvertedResidual, 32, 16, 1, 1)
        self.conv3 = _make_layers(InvertedResidual, 16, 24, 6, 2, True)
        self.conv4 = _make_layers(InvertedResidual, 24, 32, 6, 3, True)
        self.conv5 = _make_layers(InvertedResidual, 32, 64, 6, 4, True)
        self.conv6 = _make_layers(InvertedResidual, 64, 96, 6, 3)
        self.conv7 = _make_layers(InvertedResidual, 96, 160, 6, 3, True)
        self.conv8 = _make_layers(InvertedResidual, 160, 320, 6, 1)

        self.conv9 = Conv2dNormActivation(320, 1280, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)

        self.avg_pool = nn.AvgPool2d(7)
        self.conv10 = Conv2dNormActivation(1280, classes_num, 1,
                                           norm_layer=None, activation_layer=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.avg_pool(x)
        x = self.conv10(x)

        return x.view(-1, self.classes_num)


if __name__ == '__main__':
    model = MobileNetV2(classes_num=10)
    # image = torch.randn(10, 3, 224, 224)
    # result = model(image)
    #
    # print(result.shape)

    summary(model, (3, 224, 224), device="cpu")
