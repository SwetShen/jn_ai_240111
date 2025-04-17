from torch import nn
from torchvision.ops import Conv2dNormActivation


# Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, _1x1, _3x3_reduce, _3x3, _5x5_reduce, _5x5, pool_proj):
        super().__init__()
        self.b1 = Conv2dNormActivation(in_channels, _1x1, 1)
        self.b2 = nn.Sequential(
            Conv2dNormActivation(in_channels, _3x3_reduce, 1),
            Conv2dNormActivation(_3x3_reduce, _3x3, 3)
        )
        self.b3 = nn.Sequential(
            Conv2dNormActivation(in_channels, _5x5_reduce, 1),
            Conv2dNormActivation(_5x5_reduce, _5x5, 5)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            Conv2dNormActivation(in_channels, pool_proj, 1)
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        # 在通道维度上连接
        return torch.concat((b1, b2, b3, b4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.c1 = Conv2dNormActivation(3, 64, 7, stride=2)
        self.c2 = nn.Sequential(
            Conv2dNormActivation(64, 64, 1),
            Conv2dNormActivation(64, 192, 3)
        )
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(7, stride=1)

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1024, num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    # 写代码，着眼于业务
    def forward(self, x):
        # Nx3x224x224
        x = self.c1(x)
        # Nx64x112x112
        x = self.max_pool(x)
        # Nx64x56x56
        x = self.c2(x)
        # Nx192x56x56
        x = self.max_pool(x)
        # Nx192x28x28
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.max_pool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        y = self.log_softmax(x)
        return y


if __name__ == '__main__':
    import torch

    model = GoogLeNet()
    x = torch.rand(5, 3, 224, 224)
    y = model(x)
    print(y.shape)
