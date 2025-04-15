from torch import nn
from torchvision.ops import Conv2dNormActivation


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling=False):
        super().__init__()

        stride = 2 if down_sampling else 1

        # 3 x 3 卷积
        self.conv1 = Conv2dNormActivation(in_channels, out_channels,
                                          3, stride, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        # 3 x 3 卷积
        self.conv2 = Conv2dNormActivation(out_channels, out_channels,
                                          3, 1, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=None)

        # 残差分支结构：stride=1 此处为None 不做任何处理  stride=2 进行图像大小减半操作
        self.residual = Conv2dNormActivation(in_channels, out_channels,
                                             3, stride, 1,
                                             norm_layer=nn.BatchNorm2d, activation_layer=None) \
            if in_channels != out_channels or stride == 2 else None

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        # 判断是否进行图像减半的残差分支结构
        if self.residual is None:  # 没有减半操作，输入与卷积结果相加
            return self.relu(x + identity)
        else:  # 有减半操作，输入减半后 与卷积结果相加
            return self.relu(x + self.residual(identity))
