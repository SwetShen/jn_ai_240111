from torch import nn
import torch
from torchvision.ops import Conv2dNormActivation
from torchsummary import summary
from torchvision.transforms import Resize


# Depthwise Separable Convolution 深度可分离卷积
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #  深度卷积
        self.depth_wise = Conv2dNormActivation(in_channels, in_channels, 3,
                                               stride=stride, padding=1, groups=in_channels,
                                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        #  逐点卷积
        self.point_wise = Conv2dNormActivation(in_channels, out_channels, 1,
                                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


class MobileNetV1(nn.Module):
    # alpha 调整输入输出通道的参数
    # rho 调整图像输入大小参数
    def __init__(self, classes_num=1000, alpha=1., rho=1.):
        super().__init__()
        self.alpha = alpha
        self.rho = rho
        self._rho_initialize()

        self.conv1 = Conv2dNormActivation(3, self._alpha_channels(32), 3,
                                          stride=2, padding=1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.conv2 = DSConv(self._alpha_channels(32), self._alpha_channels(64))
        self.conv3 = DSConv(self._alpha_channels(64), self._alpha_channels(128), stride=2)
        self.conv4 = DSConv(self._alpha_channels(128), self._alpha_channels(128))
        self.conv5 = DSConv(self._alpha_channels(128), self._alpha_channels(256), stride=2)
        self.conv6 = DSConv(self._alpha_channels(256), self._alpha_channels(256))
        self.conv7 = DSConv(self._alpha_channels(256), self._alpha_channels(512), stride=2)
        self.conv8 = nn.Sequential(*[DSConv(self._alpha_channels(512), self._alpha_channels(512)) for _ in range(5)])
        self.conv9 = DSConv(self._alpha_channels(512), self._alpha_channels(1024), stride=2)
        self.conv10 = DSConv(self._alpha_channels(1024), self._alpha_channels(1024))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应全局池化
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self._alpha_channels(1024), classes_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def _alpha_channels(self, channels):
        return int(channels * self.alpha)

    def _rho_initialize(self):
        image_size = int(224 * self.rho)
        rest = image_size % 32
        if rest != 0:
            image_size += 32 - rest

        # 将输入图像调整成可以被32整除的值
        self.input_image_size = image_size
        self.resize_image = Resize(image_size)

    def forward(self, x):
        # 将输入的x的大小调整为可以被32整除的大小
        x = self.resize_image(x)

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

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.softmax(x)


if __name__ == '__main__':
    model = MobileNetV1(classes_num=10, alpha=1., rho=1.)
    # image = torch.randn(1, 3, 224, 224)
    # result = model(image)
    # print(result.shape)
    summary(model, (3, 224, 224), device="cpu")
