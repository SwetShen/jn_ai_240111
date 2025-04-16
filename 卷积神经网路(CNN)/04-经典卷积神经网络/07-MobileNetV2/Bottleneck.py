from torch import nn
from torchvision.ops import Conv2dNormActivation


class Bottleneck(nn.Module):
    """
    resnet中bottleneck 1x1卷积是先降维，再升维
    mobilenetv2中bottleneck 1x1卷积是先升维，再降维

    - bottleneck: 倒置残差块
    - t: 扩展因子，也就是倒置残差第一个 1x1 线性瓶颈，将原来通道放大几倍
    - c: 输出通道数 out_channels
    - n: 模块重复的次数
    - s: 步幅
    """

    def __init__(self, in_channels, out_channels, t, down_sampling=False):
        super().__init__()

        stride = 2 if down_sampling else 1

        # 1 x 1 卷积
        self.conv1 = Conv2dNormActivation(in_channels, in_channels * t, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)

        # 3 x 3 卷积 此处的需要深度卷积
        self.conv2 = Conv2dNormActivation(in_channels * t, in_channels * t,
                                          3, stride, 1, groups=in_channels * t,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)

        # 1 x 1 卷积
        self.conv3 = Conv2dNormActivation(in_channels * t, out_channels, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=None)

        # 残差分支结构：stride=1 此处为None 不做任何处理  stride=2 进行图像大小减半操作
        self.residual = Conv2dNormActivation(in_channels, out_channels,
                                             3, stride, 1,
                                             norm_layer=nn.BatchNorm2d, activation_layer=None) \
            if in_channels != out_channels or stride == 2 else None

        self.relu = nn.ReLU6()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 判断是否进行图像减半的残差分支结构
        if self.residual is None:  # 没有减半操作，输入与卷积结果相加
            return self.relu(x + identity)
        else:  # 有减半操作，输入减半后 与卷积结果相加
            return self.relu(x + self.residual(identity))
