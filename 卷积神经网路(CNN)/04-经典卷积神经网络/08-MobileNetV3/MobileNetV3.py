import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchsummary import summary


# SE :squeeze-and-excite 序列激发（注意力机制）
#   找出每个通道之间的权重关系。
class SEModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.fc2 = nn.Linear(in_channels // 4, in_channels)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.h_sig = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.h_sig(self.fc2(x))
        return x.unsqueeze(-1).unsqueeze(-1) * identity


# 倒置残差结构 = 瓶颈结构 + 深度可分离卷积
class Bottleneck(nn.Module):
    def __init__(self, kernel_size, exp_size, in_channels, out_channels, is_se, nl, stride):
        super().__init__()
        # 判断是否使用步长为2或者输入输出通道不一致的下采样
        self.is_downsample = stride == 2 or in_channels != out_channels
        # 是否使用序列激发（注意力机制）
        self.is_se = is_se

        # 激活函数选择
        self.activation = nn.ReLU6
        if nl == "HS":
            self.activation = nn.Hardswish

        self.conv1 = Conv2dNormActivation(in_channels, exp_size, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=self.activation)

        #  深度卷积
        self.conv2 = Conv2dNormActivation(exp_size, exp_size, kernel_size,
                                          stride=stride, padding=(kernel_size - 1) // 2, groups=exp_size,
                                          norm_layer=nn.BatchNorm2d, activation_layer=self.activation)
        # 序列激发模块（注意力机制）
        self.se = SEModule(exp_size)

        # 逐点卷积
        self.conv3 = Conv2dNormActivation(exp_size, out_channels, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=None)

        # 残差结构（residual）
        self.down_sampling = Conv2dNormActivation(in_channels, out_channels, 1,
                                                  stride=stride,
                                                  norm_layer=nn.BatchNorm2d, activation_layer=None)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.is_se:  # 序列激发(注意力机制)
            x = self.se(x)
        x = self.conv3(x)

        if self.is_downsample:  # 当步长为2或者输入输出通道不一致时，就会下采样
            identity = self.down_sampling(identity)
        x += identity
        return x


class MobileNetV3_Large(nn.Module):
    def __init__(self, classes_num=1000):
        super().__init__()
        self.classes_num = classes_num

        self.conv1 = Conv2dNormActivation(3, 16, 3,
                                          stride=2, padding=1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish)

        self.conv2 = Bottleneck(3, 16, 16, 16, False, "RE", 1)
        self.conv3 = Bottleneck(3, 64, 16, 24, False, "RE", 2)
        self.conv4 = Bottleneck(3, 72, 24, 24, False, "RE", 1)
        self.conv5 = Bottleneck(5, 72, 24, 40, True, "RE", 2)
        self.conv6 = Bottleneck(5, 120, 40, 40, True, "RE", 1)
        self.conv7 = Bottleneck(5, 120, 40, 40, True, "RE", 1)
        self.conv8 = Bottleneck(3, 240, 40, 80, False, "HS", 2)
        self.conv9 = Bottleneck(3, 200, 80, 80, False, "HS", 1)
        self.conv10 = Bottleneck(3, 184, 80, 80, False, "HS", 1)
        self.conv11 = Bottleneck(3, 184, 80, 80, False, "HS", 1)
        self.conv12 = Bottleneck(3, 480, 80, 112, True, "HS", 1)
        self.conv13 = Bottleneck(3, 672, 112, 112, True, "HS", 1)
        self.conv14 = Bottleneck(5, 672, 112, 160, True, "HS", 2)
        self.conv15 = Bottleneck(5, 960, 160, 160, True, "HS", 1)
        self.conv16 = Bottleneck(5, 960, 160, 160, True, "HS", 1)

        self.conv17 = Conv2dNormActivation(160, 960, 1,
                                           norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish)

        self.avg_pool = nn.AvgPool2d(7)
        self.conv18 = Conv2dNormActivation(960, 1280, 1,
                                           norm_layer=None, activation_layer=nn.Hardswish)
        self.conv19 = Conv2dNormActivation(1280, classes_num, 1,
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
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.avg_pool(x)
        x = self.conv18(x)
        x = self.conv19(x)

        return x.reshape(-1, self.classes_num)


if __name__ == '__main__':
    model = MobileNetV3_Large(classes_num=10)
    # image = torch.randn(1, 3, 224, 224)
    # result = model(image)
    # print(result.shape)

    model = model.to(torch.device("cuda"))
    summary(model, (3, 224, 224), device="cuda")
