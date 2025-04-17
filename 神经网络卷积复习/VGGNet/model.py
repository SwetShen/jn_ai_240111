from torch import nn
from torchvision.ops import Conv2dNormActivation


# 卷积块
class ConvBlock(nn.Module):
    # num_layers: 有多少个卷积层
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层卷积输入通道来自于上一层，此处单独处理
            Conv2dNormActivation(in_channels, out_channels, 3, norm_layer=None),
            *(Conv2dNormActivation(out_channels, out_channels, 3, norm_layer=None) for _ in range(num_layers - 1))
        )

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_layers(x)
        y = self.max_pool(x)
        return y


class VGGNet(nn.Module):
    # config: vgg 卷积层的配置
    # num_classes: 分类数
    def __init__(self, config=[2, 2, 3, 3, 3], num_classes=1000):
        super().__init__()
        # 5个卷积块的输出通道数
        self.out_channels = [3, 64, 128, 256, 512, 512]
        # 构造 5 个卷积块
        # 特征提取器
        self.feature_executor = nn.ModuleList([
            *(ConvBlock(self.out_channels[i], self.out_channels[i + 1], cfg) for i, cfg in enumerate(config))
        ])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(-1)
        )

    def forward(self, x):
        for conv_block in self.feature_executor:
            x = conv_block(x)
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    import torch

    model = VGGNet()
    x = torch.rand(5, 3, 224, 224)
    y = model(x)
    print(y.shape)
