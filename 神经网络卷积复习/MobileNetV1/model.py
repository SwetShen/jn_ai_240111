import math

from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms import Resize


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
    # alpha: 通道的宽度乘数，用于调节通道大小
    # rho: 分辨率乘数，用于调节图片的大小
    def __init__(self, num_classes=1000, alpha=1., rho=1.):
        super(MobileNetV1, self).__init__()
        self.alpha = alpha
        self.rho = rho
        # 应用了分辨率乘数的图片大小
        self.input_img_size = int(224 * self.rho)
        self.c1 = Conv2dNormActivation(3, self._alpha_channel_num(32), 3, stride=2)
        self.dsc1 = DepthwiseSeparableConv(self._alpha_channel_num(32), self._alpha_channel_num(64), stride=1)
        self.dsc2 = DepthwiseSeparableConv(self._alpha_channel_num(64), self._alpha_channel_num(128), stride=2)
        self.dsc3 = DepthwiseSeparableConv(self._alpha_channel_num(128), self._alpha_channel_num(128), stride=1)
        self.dsc4 = DepthwiseSeparableConv(self._alpha_channel_num(128), self._alpha_channel_num(256), stride=2)
        self.dsc5 = DepthwiseSeparableConv(self._alpha_channel_num(256), self._alpha_channel_num(256), stride=1)
        self.dsc6 = DepthwiseSeparableConv(self._alpha_channel_num(256), self._alpha_channel_num(512), stride=2)
        self.dsc7 = nn.Sequential(
            *(DepthwiseSeparableConv(self._alpha_channel_num(512), self._alpha_channel_num(512), stride=1) for _ in
              range(5))
        )
        self.dsc8 = DepthwiseSeparableConv(self._alpha_channel_num(512), self._alpha_channel_num(1024), stride=2)
        # 输入 dsc9 时，图片的大小
        input_size = self.input_img_size / 32
        p = math.ceil((input_size + 1) / 2)
        self.dsc9 = nn.Sequential(
            Conv2dNormActivation(self._alpha_channel_num(1024), self._alpha_channel_num(1024), 3, stride=2, padding=p,
                                 groups=self._alpha_channel_num(1024)),
            Conv2dNormActivation(self._alpha_channel_num(1024), self._alpha_channel_num(1024), 1)
        )
        # self.avg_pool = nn.AvgPool2d(7)
        # 自适应池化: 池化核大小会自动调整，只需要填输出形状即可
        # 参数 1 代表输出图像大小为 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self._alpha_channel_num(1024), num_classes),
            nn.LogSoftmax(dim=-1)
        )

    # 通道数乘以 alpha 系数后取整的函数
    def _alpha_channel_num(self, channels):
        return int(channels * self.alpha)

    # 将图片大小根据 rho 乘数进行调整
    def _rho_image_transform(self, x):
        return Resize((self.input_img_size, self.input_img_size), antialias=True)(x)

    def forward(self, x):
        # 调整图像大小
        x = self._rho_image_transform(x)
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

    model = MobileNetV1(alpha=2, rho=2)
    # x = torch.rand(5, 3, 224, 224)
    x = torch.rand(5, 3, 320, 320)
    print(model(x).shape)
