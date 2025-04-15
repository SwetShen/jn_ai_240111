import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from BasicBlock import BasicBlock  # 基础的残差块
from Bottleneck import Bottleneck  # 瓶颈层残差块


def _make_layers(in_channels, out_channels, block, num_layers, down_sampling=False):
    return nn.Sequential(*[block(in_channels, out_channels, down_sampling) if i == 0 else
                           block(out_channels, out_channels, False) for i in range(num_layers)])


class ResNet(nn.Module):
    def __init__(self, block, num_classes=1000, layers=(2, 2, 2, 2)):
        super().__init__()
        self.conv1 = Conv2dNormActivation(3, 64,
                                          7, 2, 3,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.max_pool = nn.MaxPool2d(3, 2, 1)

        is_bottleneck = block == Bottleneck
        in_channels = 64
        out_channels = 256 if is_bottleneck else 64

        self.conv2 = _make_layers(in_channels, out_channels, block, layers[0])
        self.conv3 = _make_layers(in_channels * 2, out_channels * 2, block, layers[0], True)
        self.conv4 = _make_layers(in_channels * 2 ** 2, out_channels * 2 ** 2, block, layers[0], True)
        self.conv5 = _make_layers(in_channels * 2 ** 3, out_channels * 2 ** 3, block, layers[0], True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels * 2 ** 3, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.softmax(self.fc(x))
        return x


def resnet18(classes_num=1000):
    return ResNet(BasicBlock, classes_num, [2, 2, 2, 2])


def resnet34(classes_num=1000):
    return ResNet(BasicBlock, classes_num, [3, 4, 6, 3])


def resnet50(classes_num=1000):
    return ResNet(Bottleneck, classes_num, [3, 4, 6, 3])


def resnet101(classes_num=1000):
    return ResNet(Bottleneck, classes_num, [3, 4, 23, 3])


def resnet152(classes_num=1000):
    return ResNet(Bottleneck, classes_num, [3, 8, 36, 3])
